import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from geometry_utils import GeometryEngine


class ArmVisualizer:
    def __init__(self, robot_arm):
        self.arm = robot_arm
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d', proj_type='persp')

        self.collision_locked = False

        self.num_dof = len(self.arm.modules)
        self.default_states = [m['L'] if m['type'] == 'A' else 0.0 for m in self.arm.modules]
        self.last_valid_states = list(self.default_states)
        self.target_states = list(self.default_states)
        self.current_time = 0.0

        self.trajectory = []
        self.surfaces = []
        self.axis_lines = []
        self.torque_texts = []
        self.traj_line = None

        self.slider_ranges = []
        self.sliders = []
        self.slider_current_fills = []

        self.ax.set_xlim([-2, 10])
        self.ax.set_ylim([-2, 10])
        self.ax.set_zlim([0, 12])
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._set_normal_title()

        plt.subplots_adjust(bottom=0.1 + 0.04 * self.num_dof)
        for i, m in enumerate(self.arm.modules):
            ax_s = plt.axes([0.2, 0.05 + i * 0.035, 0.6, 0.02])
            v_max = m['L'] if m['type'] == 'A' else np.pi
            v_min = 0 if m['type'] == 'A' else -np.pi
            slider = Slider(ax_s, f"{m['type']}{i+1}", v_min, v_max, valinit=self.target_states[i])
            slider.poly.set_facecolor('#4c72b0')
            slider.poly.set_alpha(0.12)
            slider.on_changed(lambda _: self._on_slider_move())
            self.sliders.append(slider)
            self.slider_ranges.append((v_min, v_max))
            self.slider_current_fills.append(None)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._block_event = False
        self._update_slider_visuals()
        self.render()

    def _clip_joint_value(self, index, value):
        module = self.arm.modules[index]
        if module['type'] == 'A':
            return float(np.clip(value, 0.0, module['L']))
        return float(np.clip(value, -np.pi, np.pi))

    def _step_towards_target(self, dt):
        next_states = list(self.last_valid_states)
        moved = False
        for i, (curr, target) in enumerate(zip(self.last_valid_states, self.target_states)):
            delta = target - curr
            if abs(delta) < 1e-9:
                continue

            max_step = max(0.0, self.arm.modules[i]['speed']) * dt
            if max_step <= 0:
                continue

            step = np.clip(delta, -max_step, max_step)
            next_states[i] = curr + step
            moved = moved or abs(step) > 1e-9
        return next_states, moved

    def _update_slider_visuals(self):
        for i, slider in enumerate(self.sliders):
            old_fill = self.slider_current_fills[i]
            if old_fill is not None:
                old_fill.remove()

            v_min, _ = self.slider_ranges[i]
            curr = self.last_valid_states[i]
            target = self.target_states[i]
            self.slider_current_fills[i] = slider.ax.axvspan(
                v_min,
                curr,
                0.05,
                0.95,
                facecolor='#66bb6a',
                alpha=0.35,
                zorder=0,
            )
            slider.valtext.set_text(f"C:{curr:.2f} T:{target:.2f}")

    def _set_slider_targets(self):
        self._block_event = True
        for slider, target in zip(self.sliders, self.target_states):
            slider.set_val(target)
        self._block_event = False
        self._update_slider_visuals()

    def _set_normal_title(self):
        self.ax.set_title(f"t = {self.current_time:.2f}s | 'C' Clear Trajectory", color='black')

    def _set_collision_title(self):
        self.ax.set_title(
            f"t = {self.current_time:.2f}s | !!! COLLISION !!!\nPress 'R' to Reset System",
            color='red',
            fontweight='bold',
        )

    def _on_slider_move(self):
        if self._block_event:
            return
        if self.collision_locked:
            return

        self.target_states = [self._clip_joint_value(i, slider.val) for i, slider in enumerate(self.sliders)]
        self._update_slider_visuals()
        self.fig.canvas.draw_idle()

    def _lock_system(self):
        self.collision_locked = True
        self._set_collision_title()
        self.fig.canvas.draw_idle()

    def _reset_all(self):
        self.collision_locked = False
        self.last_valid_states = list(self.default_states)
        self.target_states = list(self.default_states)
        self.trajectory = []
        if self.traj_line:
            self.traj_line.remove()
            self.traj_line = None

        self._set_slider_targets()
        self._set_normal_title()
        self.render()

    def set_joint_states(self, states, dt, current_time=None):
        """ External control interface. Updates targets, then moves current states by speed limit. """
        if self.collision_locked:
            return

        if current_time is not None:
            self.current_time = float(current_time)
            if self.collision_locked:
                self._set_collision_title()
            else:
                self._set_normal_title()

        new_targets = list(self.target_states)
        if isinstance(states, dict):
            for k, v in states.items():
                if v is not None:
                    new_targets[k] = self._clip_joint_value(k, v)
        else:
            for i, v in enumerate(states):
                if v is not None:
                    new_targets[i] = self._clip_joint_value(i, v)

        targets_changed = any(abs(a - b) > 1e-9 for a, b in zip(new_targets, self.target_states))
        self.target_states = new_targets
        if targets_changed:
            self._set_slider_targets()
        else:
            self._update_slider_visuals()

        dt = min(max(float(dt), 0.0), 0.2)
        new_states, moved = self._step_towards_target(dt)
        if not moved:
            self.fig.canvas.draw_idle()
            return

        if self.arm.check_collision(new_states, self.last_valid_states):
            self._lock_system()
            return

        self.last_valid_states = new_states
        self._update_slider_visuals()
        self.render()

    def render(self):
        for obj in self.surfaces + self.axis_lines + self.torque_texts:
            try:
                obj.remove()
            except Exception:
                pass
        self.surfaces, self.axis_lines, self.torque_texts = [], [], []

        segs, joints, servo_frames, joint_torques = self.arm.forward_kinematics(self.last_valid_states)

        axis_len = 1.2
        for frame in servo_frames:
            p0 = frame['pos']
            p_axis = p0 + frame['axis'] * axis_len
            l1, = self.ax.plot(
                [p0[0], p_axis[0]],
                [p0[1], p_axis[1]],
                [p0[2], p_axis[2]],
                color='red',
                linewidth=3,
                zorder=10,
            )
            self.axis_lines.append(l1)

            if frame.get('zero') is not None:
                p_zero = p0 + frame['zero'] * axis_len
                l2, = self.ax.plot(
                    [p0[0], p_zero[0]],
                    [p0[1], p_zero[1]],
                    [p0[2], p_zero[2]],
                    color='green',
                    linewidth=3,
                    zorder=10,
                )
                self.axis_lines.append(l2)

        for seg in segs:
            mesh = GeometryEngine.get_cylinder_mesh(seg['start'], seg['end'], seg['radius'])
            if mesh:
                color = 'red' if seg['type'] == 'A' else '#1f77b4'
                self.surfaces.append(
                    self.ax.plot_surface(*mesh, color=color, shade=True, antialiased=False, alpha=0.9)
                )

        overload_indices = {load['module_index'] for load in joint_torques if load['overloaded']}
        for idx, joint in enumerate(joints[:-1]):
            mesh = GeometryEngine.get_sphere_mesh(joint, 0.2)
            joint_color = 'red' if idx in overload_indices else 'gold'
            self.surfaces.append(self.ax.plot_surface(*mesh, color=joint_color, shade=True, antialiased=False))

        end_pt = joints[-1]
        if not self.trajectory or np.linalg.norm(self.trajectory[-1] - end_pt) > 0.05:
            self.trajectory.append(end_pt.copy())
        if len(self.trajectory) > 1:
            t_pts = np.array(self.trajectory)
            if self.traj_line:
                self.traj_line.set_data_3d(t_pts[:, 0], t_pts[:, 1], t_pts[:, 2])
            else:
                self.traj_line, = self.ax.plot(t_pts[:, 0], t_pts[:, 1], t_pts[:, 2], 'k--', lw=1, alpha=0.3)

        mesh_end = GeometryEngine.get_sphere_mesh(joints[-1], 0.25)
        self.surfaces.append(self.ax.plot_surface(*mesh_end, color='lime', shade=True, antialiased=False))

        for load in joint_torques:
            text = f"{load['type']}{load['module_index'] + 1} |M|={load['torque_magnitude']:.1f}Nm"
            if load['type'] == 'B':
                text += f"\n|Taxis|={load['axis_torque']:.1f}Nm"
            text += f"\nlimit={load['max_torque']:.1f}Nm"
            box_face = '#ffdddd' if load['overloaded'] else 'white'
            box_edge = '#cc0000' if load['overloaded'] else '#666666'
            text_color = '#aa0000' if load['overloaded'] else 'black'
            label = self.ax.text(
                load['label_pos'][0],
                load['label_pos'][1],
                load['label_pos'][2],
                text,
                fontsize=8,
                color=text_color,
                ha='left',
                va='bottom',
                bbox={
                    'boxstyle': 'round,pad=0.25',
                    'facecolor': box_face,
                    'alpha': 0.85,
                    'edgecolor': box_edge,
                },
            )
            self.torque_texts.append(label)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if not event.key:
            return
        key = event.key.lower()
        if key == 'c':
            self.trajectory = []
            if self.traj_line:
                self.traj_line.remove()
                self.traj_line = None
            self.render()
        elif key == 'r':
            self._reset_all()
