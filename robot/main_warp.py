import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from robot_model import RobotArm
from visual import ArmVisualizer
from warp_ik import WarpPoseIKSolver


def build_warp_solver(args, robot):
    return WarpPoseIKSolver(
        robot,
        device=args.device,
        step_size=args.step_size,
        pos_weight=args.pose_pos_weight,
        dir_weight=args.pose_dir_weight,
        damp=args.damp,
        num_restarts=args.num_restarts,
        top_k_seeds=args.top_k_seeds,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warp-based differentiable IK demo.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--pose-pos-weight", type=float, default=1.0)
    parser.add_argument("--pose-dir-weight", type=float, default=1.0)
    parser.add_argument("--damp", type=float, default=0.05)
    parser.add_argument("--num-restarts", type=int, default=2048)
    parser.add_argument("--top-k-seeds", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    # A [方向n] [行程L] [杆件质量] [下一模组质量] [最大力矩] [速度]
    # B [旋转轴n1] [零位朝向n2] [杆长L] [杆件质量] [下一模组质量] [最大力矩] [速度]
    # 最后一个模组质量即为末端负载质量
    config = """
    B 0 0 1  1 0 0  3  0.6 0.9 35 1.2
    B 0 1 0  1 0 0  3  0.6 0.9 35 1.2
    B 0 1 0  1 0 0  3  0.6 0.9 35 1.2
    B 0 1 0  1 0 0  1  0.4 0.7 18 0.9
    """

    robot = RobotArm(config)
    solver = build_warp_solver(args, robot)
    viz = ArmVisualizer(robot)

    apple_pos = np.array([1.0, 3.2, 5.0], dtype=np.float32)
    grab_dir = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    current_q = np.zeros(len(robot.modules), dtype=np.float32)
    target_q = solver.solve(apple_pos, grab_dir, current_q, max_iter=args.max_iter, tol=args.tol)

    plt.ion()

    t = 0.0
    last_wall_time = time.perf_counter()
    while plt.fignum_exists(viz.fig.number):
        now_wall_time = time.perf_counter()
        dt = min(max(now_wall_time - last_wall_time, 1e-4), 0.2)
        last_wall_time = now_wall_time
        t += dt

        if not robot.collision_locked:
            grab_dir = np.array([np.cos(0.5 * t), np.sin(0.5 * t), 1.0], dtype=np.float32)

        pee, zee = robot.get_ee_pose()
        if np.linalg.norm(pee - apple_pos) > 0.1 or np.linalg.norm(zee - grab_dir) > 0.1:
            print("new apple")
            current_q = robot.get_current_q()
            target_q = solver.solve(apple_pos, grab_dir, current_q, max_iter=args.max_iter, tol=args.tol)

        robot.set_target(target_q)
        status = robot.update(dt)
        if status == "COLLISION":
            print(f"检测到碰撞！t={t:.2f}")

        viz.set_target_visual(apple_pos, grab_dir)
        viz.render(t)
        plt.pause(0.001)
