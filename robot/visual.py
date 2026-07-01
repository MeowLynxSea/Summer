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
        
        self.trajectory = []
        self.surfaces = []
        self.axis_lines = []
        self.traj_line = None 

        self.ax.set_xlim([-2, 10]); self.ax.set_ylim([-2, 10]); self.ax.set_zlim([0, 12])
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")
        self._set_normal_title()

        plt.subplots_adjust(bottom=0.1 + 0.04 * self.num_dof)
        self.sliders = []
        for i, m in enumerate(self.arm.modules):
            ax_s = plt.axes([0.2, 0.05 + i * 0.035, 0.6, 0.02])
            v_max = m['L'] if m['type'] == 'A' else np.pi
            v_min = 0 if m['type'] == 'A' else -np.pi
            s = Slider(ax_s, f"{m['type']}{i+1}", v_min, v_max, valinit=self.last_valid_states[i])
            s.on_changed(lambda _: self._on_slider_move())
            self.sliders.append(s)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._block_event = False 
        self.render()

    def _set_normal_title(self):
        self.ax.set_title("'C' Clear Trajectory", color='black')

    def _set_collision_title(self):
        self.ax.set_title("!!! COLLISION !!!\nPress 'R' to Reset System", color='red', fontweight='bold')

    def _on_slider_move(self):
        if self._block_event: return
        if self.collision_locked: return # 锁定状态下禁止通过滑块改变

        curr_states = [s.val for s in self.sliders]
        if self.arm.check_collision(curr_states, self.last_valid_states):
            self._lock_system()
        else:
            self.last_valid_states = list(curr_states)
            self.render()

    def _lock_system(self):
        self.collision_locked = True
        self._set_collision_title()
        self.fig.canvas.draw_idle()

    def _reset_all(self):
        self.collision_locked = False
        self._block_event = True
        for s, val in zip(self.sliders, self.default_states):
            s.set_val(val)
        self.last_valid_states = list(self.default_states)
        self.trajectory = []
        if self.traj_line: 
            self.traj_line.remove()
            self.traj_line = None
        self._block_event = False
        
        self._set_normal_title()
        self.render()

    def set_joint_states(self, states):
        """ 外部算法控制接口 """
        if self.collision_locked: 
            return 

        new_states = list(self.last_valid_states)
        if isinstance(states, dict):
            for k, v in states.items(): new_states[k] = v
        else:
            for i, v in enumerate(states):
                if v is not None: new_states[i] = v
        
        if self.arm.check_collision(new_states, self.last_valid_states):
            self._lock_system()
        else:
            self._block_event = True
            for i, v in enumerate(new_states):
                self.sliders[i].set_val(v)
            self.last_valid_states = new_states
            self._block_event = False
            self.render()

    def render(self):
        for obj in self.surfaces + self.axis_lines:
            try: obj.remove()
            except: pass
        self.surfaces, self.axis_lines = [], []
        
        segs, joints, servo_frames = self.arm.forward_kinematics(self.last_valid_states)
        
        axis_len = 1.2
        for frame in servo_frames:
            p0 = frame['pos']
            p_axis = p0 + frame['axis'] * axis_len
            l1, = self.ax.plot([p0[0], p_axis[0]], [p0[1], p_axis[1]], [p0[2], p_axis[2]], color='red', linewidth=3, zorder=10)

            p_zero = p0 + frame['zero'] * axis_len
            l2, = self.ax.plot([p0[0], p_zero[0]], [p0[1], p_zero[1]], [p0[2], p_zero[2]], color='green', linewidth=3, zorder=10)
            self.axis_lines.extend([l1, l2])

        # 机械臂
        for s in segs:
            mesh = GeometryEngine.get_cylinder_mesh(s['start'], s['end'], s['radius'])
            if mesh:
                c = 'red' if s['type'] == 'A' else '#1f77b4'
                self.surfaces.append(self.ax.plot_surface(*mesh, color=c, shade=True, antialiased=False, alpha=0.9))
        
        for j in joints[:-1]:
            mesh = GeometryEngine.get_sphere_mesh(j, 0.2)
            self.surfaces.append(self.ax.plot_surface(*mesh, color='gold', shade=True, antialiased=False))

        # 轨迹线
        end_pt = joints[-1]
        if not self.trajectory or np.linalg.norm(self.trajectory[-1] - end_pt) > 0.05:
            self.trajectory.append(end_pt.copy())
        if len(self.trajectory) > 1:
            t_pts = np.array(self.trajectory)
            if self.traj_line: self.traj_line.set_data_3d(t_pts[:,0], t_pts[:,1], t_pts[:,2])
            else: self.traj_line, = self.ax.plot(t_pts[:,0], t_pts[:,1], t_pts[:,2], 'k--', lw=1, alpha=0.3)

        # 末端
        mesh_end = GeometryEngine.get_sphere_mesh(joints[-1], 0.25)
        surf_end = self.ax.plot_surface(*mesh_end, color='lime', shade=True, antialiased=False)
        self.surfaces.append(surf_end)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if not event.key: return
        key = event.key.lower()
        if key == 'c':
            self.trajectory = []
            if self.traj_line: 
                self.traj_line.remove()
                self.traj_line = None
            self.render()
        elif key == 'r':
            self._reset_all()