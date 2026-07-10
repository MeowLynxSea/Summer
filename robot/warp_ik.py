from contextlib import nullcontext

import numpy as np

from robot_model import RobotArm

import warp as wp


@wp.func
def _safe_normalize(v: wp.vec3):
    n = wp.length(v)
    if n > 1.0e-8:
        return v / n
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def _rotate_vector(v: wp.vec3, axis: wp.vec3, theta: float):
    axis_n = _safe_normalize(axis)
    c = wp.cos(theta)
    s = wp.sin(theta)
    return v * c + wp.cross(axis_n, v) * s + axis_n * wp.dot(axis_n, v) * (1.0 - c)


@wp.func
def _transform_local(local_v: wp.vec3, basis_x: wp.vec3, basis_y: wp.vec3, basis_z: wp.vec3):
    return basis_x * local_v[0] + basis_y * local_v[1] + basis_z * local_v[2]


@wp.kernel
def _forward_kinematics_kernel(
    num_modules: int,
    module_types: wp.array(dtype=wp.int32),
    n1s: wp.array(dtype=wp.vec3),
    n2s: wp.array(dtype=wp.vec3),
    lengths: wp.array(dtype=wp.float32),
    q: wp.array(dtype=wp.float32),
    ee_pos: wp.array(dtype=wp.vec3),
    ee_dir: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid != 0:
        return

    p = wp.vec3(0.0, 0.0, 0.0)
    basis_x = wp.vec3(1.0, 0.0, 0.0)
    basis_y = wp.vec3(0.0, 1.0, 0.0)
    basis_z = wp.vec3(0.0, 0.0, 1.0)
    z_ee = wp.vec3(0.0, 0.0, 1.0)

    for i in range(num_modules):
        joint_type = module_types[i]
        n1_local = _safe_normalize(n1s[i])

        if joint_type == 0:
            d_glob = _transform_local(n1_local, basis_x, basis_y, basis_z)
            p = p + d_glob * q[i]
            z_ee = d_glob
        else:
            axis_glob = _transform_local(n1_local, basis_x, basis_y, basis_z)
            theta = q[i]
            basis_x = _rotate_vector(basis_x, axis_glob, theta)
            basis_y = _rotate_vector(basis_y, axis_glob, theta)
            basis_z = _rotate_vector(basis_z, axis_glob, theta)

            n2_local = _safe_normalize(n2s[i])
            v_rod = _transform_local(n2_local, basis_x, basis_y, basis_z)
            p = p + v_rod * lengths[i]
            z_ee = v_rod

    ee_pos[0] = p
    ee_dir[0] = _safe_normalize(z_ee)


@wp.kernel
def _pose_loss_kernel(
    target_pos: wp.vec3,
    target_dir: wp.vec3,
    pos_weight: float,
    dir_weight: float,
    ee_pos: wp.array(dtype=wp.vec3),
    ee_dir: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid != 0:
        return

    pos_err = ee_pos[0] - target_pos
    dir_err = _safe_normalize(ee_dir[0]) - _safe_normalize(target_dir)
    loss[0] = pos_weight * wp.dot(pos_err, pos_err) + dir_weight * wp.dot(dir_err, dir_err)


class WarpPoseIKSolver:
    def __init__(
        self,
        robot: RobotArm,
        device="auto",
        step_size=1.0,
        pos_weight=1.0,
        dir_weight=1.0,
        damp=0.05,
        num_restarts=12,
        top_k_seeds=4,
        random_seed=0,
    ):
        self.robot = robot
        self.step_size = float(step_size)
        self.pos_weight = np.float32(pos_weight)
        self.dir_weight = np.float32(dir_weight)
        self.damp = float(damp)
        self.num_restarts = int(num_restarts)
        self.top_k_seeds = int(top_k_seeds)
        self.rng = np.random.default_rng(random_seed)

        wp.init()
        self.device = self._resolve_device(device)

        module_types = []
        n1s = []
        n2s = []
        lengths = []
        q_min = []
        q_max = []
        for m in robot.modules:
            is_prismatic = 0 if m["type"] == "A" else 1
            module_types.append(is_prismatic)
            n1 = np.asarray(m["n1"], dtype=np.float32)
            n1_norm = np.linalg.norm(n1)
            if n1_norm > 1e-8:
                n1 = n1 / n1_norm
            n1s.append(n1)

            if m["type"] == "B":
                n2 = np.asarray(m["n2"], dtype=np.float32)
                n2_norm = np.linalg.norm(n2)
                if n2_norm > 1e-8:
                    n2 = n2 / n2_norm
                n2s.append(n2)
                q_min.append(-np.pi)
                q_max.append(np.pi)
            else:
                n2s.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                q_min.append(0.0)
                q_max.append(m["L"])

            lengths.append(np.float32(m["L"]))

        self.num_modules = len(robot.modules)
        self.q_min = np.asarray(q_min, dtype=np.float32)
        self.q_max = np.asarray(q_max, dtype=np.float32)
        self.module_types_wp = wp.array(np.asarray(module_types, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.n1s_wp = wp.array(np.asarray(n1s, dtype=np.float32), dtype=wp.vec3, device=self.device)
        self.n2s_wp = wp.array(np.asarray(n2s, dtype=np.float32), dtype=wp.vec3, device=self.device)
        self.lengths_wp = wp.array(np.asarray(lengths, dtype=np.float32), dtype=wp.float32, device=self.device)
        eye = np.eye(3, dtype=np.float32)
        self.seed_vecs_wp = [wp.array(eye[i : i + 1], dtype=wp.vec3, device=self.device) for i in range(3)]

    def _resolve_device(self, device):
        if device in (None, "auto"):
            return "cuda:0" if wp.is_cuda_available() else "cpu"
        return device

    def _forward(self, q, requires_grad=False):
        q_arr = wp.array(q.astype(np.float32), dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        ee_pos = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        ee_dir = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        tape = wp.Tape() if requires_grad else None
        context = tape if tape is not None else nullcontext()
        with context:
            wp.launch(
                kernel=_forward_kinematics_kernel,
                dim=1,
                inputs=[
                    self.num_modules,
                    self.module_types_wp,
                    self.n1s_wp,
                    self.n2s_wp,
                    self.lengths_wp,
                    q_arr,
                ],
                outputs=[ee_pos, ee_dir],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        return q_arr, ee_pos, ee_dir, tape

    def _pose_residual(self, p_ee, z_ee, target_pos, target_dir):
        err_p = target_pos - p_ee
        err_o = np.cross(z_ee, target_dir)
        pos_scale = np.sqrt(self.pos_weight)
        dir_scale = np.sqrt(self.dir_weight)
        residual = np.concatenate([pos_scale * err_p, dir_scale * err_o]).astype(np.float32)
        loss = float(np.dot(residual, residual))
        return loss, err_p.astype(np.float32), err_o.astype(np.float32), residual

    def _forward_and_jacobian(self, q, target_pos, target_dir):
        q_arr, ee_pos, ee_dir, tape = self._forward(q, requires_grad=True)
        p_ee = ee_pos.numpy()[0].astype(np.float32)
        z_ee = ee_dir.numpy()[0].astype(np.float32)

        j_pos = np.zeros((3, self.num_modules), dtype=np.float32)
        j_dir = np.zeros((3, self.num_modules), dtype=np.float32)
        for i in range(3):
            tape.backward(grads={ee_pos: self.seed_vecs_wp[i]})
            j_pos[i] = q_arr.grad.numpy().astype(np.float32)
            tape.zero()

        for i in range(3):
            tape.backward(grads={ee_dir: self.seed_vecs_wp[i]})
            j_dir[i] = q_arr.grad.numpy().astype(np.float32)
            tape.zero()

        loss, err_p, err_o, residual = self._pose_residual(p_ee, z_ee, target_pos, target_dir)
        j_angular = np.zeros_like(j_dir)
        for j in range(self.num_modules):
            # For unit direction z, dz/dq = omega x z, so omega = z x dz/dq.
            j_angular[:, j] = np.cross(z_ee, j_dir[:, j])

        pos_scale = np.sqrt(self.pos_weight)
        dir_scale = np.sqrt(self.dir_weight)
        jacobian = np.vstack([pos_scale * j_pos, dir_scale * j_angular]).astype(np.float32)
        return loss, p_ee, z_ee, err_p, err_o, residual, jacobian

    def _sample_initial_guesses(self, q_init):
        candidates = [np.clip(np.asarray(q_init, dtype=np.float32), self.q_min, self.q_max)]
        q_mid = ((self.q_min + self.q_max) * 0.5).astype(np.float32)
        candidates.append(q_mid)
        for _ in range(max(self.num_restarts - len(candidates), 0)):
            alpha = self.rng.uniform(size=self.num_modules).astype(np.float32)
            candidates.append(self.q_min + alpha * (self.q_max - self.q_min))
        scored = []
        for q in candidates:
            _, ee_pos, ee_dir, _ = self._forward(q, requires_grad=False)
            p_ee = ee_pos.numpy()[0].astype(np.float32)
            z_ee = ee_dir.numpy()[0].astype(np.float32)
            loss, _, _, _ = self._pose_residual(p_ee, z_ee, self._target_pos, self._target_dir)
            scored.append((loss, q.copy()))
        scored.sort(key=lambda item: item[0])
        keep = max(1, min(self.top_k_seeds, len(scored)))
        return [item[1] for item in scored[:keep]]

    def _local_solve(self, q_start, max_iter, tol):
        q = q_start.copy()
        best_q = q.copy()
        best_loss = float("inf")
        damping = self.damp

        for i in range(max_iter):
            loss, p_ee, z_ee, err_p, err_o, residual, jacobian = self._forward_and_jacobian(
                q, self._target_pos, self._target_dir
            )
            if loss < best_loss:
                best_loss = loss
                best_q = q.copy()

            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_o) < tol:
                print(f"Warp IK 收敛于第 {i} 次迭代, loss={loss:.6f}")
                return q, loss

            jt = jacobian.T
            a = jt @ jacobian + (damping ** 2) * np.eye(self.num_modules, dtype=np.float32)
            try:
                dq = np.linalg.solve(a, jt @ residual)
            except np.linalg.LinAlgError:
                break

            accepted = False
            for scale in (self.step_size, 0.5 * self.step_size, 0.25 * self.step_size, 0.1 * self.step_size):
                candidate = np.clip(q + scale * dq, self.q_min, self.q_max)
                _, ee_pos, ee_dir, _ = self._forward(candidate, requires_grad=False)
                cand_p = ee_pos.numpy()[0].astype(np.float32)
                cand_z = ee_dir.numpy()[0].astype(np.float32)
                cand_loss, _, _, _ = self._pose_residual(cand_p, cand_z, self._target_pos, self._target_dir)
                if cand_loss < loss:
                    q = candidate
                    damping = max(1e-4, damping * 0.7)
                    accepted = True
                    break

            if not accepted:
                damping = min(1.0, damping * 2.0)
                if damping >= 1.0:
                    break

        return best_q, best_loss

    def solve(self, target_pos, target_n, q_init, max_iter=200, tol=1e-4):
        target_n = np.asarray(target_n, dtype=np.float32)
        target_n_norm = np.linalg.norm(target_n)
        if target_n_norm < 1e-8:
            raise ValueError("target_n must be non-zero")

        self._target_pos = np.asarray(target_pos, dtype=np.float32)
        self._target_dir = target_n / target_n_norm

        best_q = np.clip(np.asarray(q_init, dtype=np.float32), self.q_min, self.q_max)
        best_loss = float("inf")
        for seed_q in self._sample_initial_guesses(q_init):
            q_candidate, loss = self._local_solve(seed_q, max_iter=max_iter, tol=tol)
            if loss < best_loss:
                best_loss = loss
                best_q = q_candidate.copy()

        print(f"Warp IK 未收敛，返回最好结果 loss={best_loss:.6f}")
        return best_q


def solve_pose_with_warp(robot, target_pos, target_n, q_init, **solver_kwargs):
    solver = WarpPoseIKSolver(robot, **solver_kwargs)
    return solver.solve(target_pos, target_n, q_init)
