import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from geometry_utils import GeometryEngine

class ArmVisualizer:
    def __init__(self, robot_arm):
      
        self.arm = robot_arm
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d', proj_type='persp')

        self.num_dof = len(self.arm.modules)
        
        self.trajectory = []
        self.surfaces = []
        self.axis_lines = []
        self.torque_texts = []
        self.traj_line = None

        self.target_marker_pos = None
        self.target_marker_dir = None

        self.ax.set_xlim([-2, 10])
        self.ax.set_ylim([-2, 10])
        self.ax.set_zlim([0, 12])
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.sliders = []
        self.slider_current_fills = []
        self._setup_ui()

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._block_event = False 

        self.render()

    def _setup_ui(self):
        plt.subplots_adjust(bottom=0.1 + 0.04 * self.num_dof)
        for i, m in enumerate(self.arm.modules):
            ax_s = plt.axes([0.2, 0.05 + i * 0.035, 0.6, 0.02])
            v_min = self.arm.q_min[i]
            v_max = self.arm.q_max[i]
            
            slider = Slider(ax_s, f"{m['type']}{i+1}", v_min, v_max, valinit=self.arm.q_current[i])
            slider.poly.set_facecolor('#4c72b0')
            slider.poly.set_alpha(0.12)
            
            slider.on_changed(lambda _, idx=i: self._on_slider_move(idx))
            
            self.sliders.append(slider)
            self.slider_current_fills.append(None)

    def set_target_visual(self, pos, direction=None):
        self.target_marker_pos = np.array(pos) if pos is not None else None
        if direction is not None:
            self.target_marker_dir = np.array(direction) / np.linalg.norm(direction)
        else:
            self.target_marker_dir = None

    def _on_slider_move(self, index):
        if self._block_event:
            return
        
        new_targets = [s.val for s in self.sliders]
        self.arm.set_target(new_targets)
        self.fig.canvas.draw_idle()

    def _sync_ui(self):
        self._block_event = True
        for i, slider in enumerate(self.sliders):
            curr = self.arm.q_current[i]
            target = self.arm.q_target
            
            slider.set_val(target[i])
            
            if self.slider_current_fills[i] is not None:
                self.slider_current_fills[i].remove()
            
            v_min = self.arm.q_min[i]
            self.slider_current_fills[i] = slider.ax.axvspan(
                v_min, curr, 0.05, 0.95, facecolor='#66bb6a', alpha=0.35, zorder=0
            )
            
            slider.valtext.set_text(f"C:{curr:.2f} T:{target[i]:.2f}")
        self._block_event = False

    def _set_titles(self, t):
        if self.arm.collision_locked:
            self.ax.set_title(
                f"t = {t:.2f}s | !!! COLLISION LOCKED !!!\nPress 'R' to Reset",
                color='red', fontweight='bold'
            )
        else:
            self.ax.set_title(f"t = {t:.2f}s | 'C' Clear Traj | 'R' Reset", color='black')

    def render(self, t=0.0):
        for obj in self.surfaces + self.axis_lines + self.torque_texts:
            try: obj.remove()
            except: pass
        self.surfaces, self.axis_lines, self.torque_texts = [], [], []

        self._sync_ui()
        self._set_titles(t)

        curr_q = self.arm.q_current
        segs, joints, servo_frames, joint_torques = self.arm.forward_kinematics(curr_q)

        axis_len = 1.2
        for frame in servo_frames:
            p0 = frame['pos']
            p_axis = p0 + frame['axis'] * axis_len
            l1, = self.ax.plot([p0[0], p_axis[0]], [p0[1], p_axis[1]], [p0[2], p_axis[2]], 
                              color='red', linewidth=2, zorder=10)
            self.axis_lines.append(l1)
            if frame.get('zero') is not None:
                p_zero = p0 + frame['zero'] * axis_len
                l2, = self.ax.plot([p0[0], p_zero[0]], [p0[1], p_zero[1]], [p0[2], p_zero[2]], 
                                  color='green', linewidth=2, zorder=10, alpha=0.6)
                self.axis_lines.append(l2)

        for seg in segs:
            mesh = GeometryEngine.get_cylinder_mesh(seg['start'], seg['end'], seg['radius'])
            if mesh:
                color = '#ff4444' if seg['type'] == 'A' else '#1f77b4'
                self.surfaces.append(self.ax.plot_surface(*mesh, color=color, shade=True, alpha=0.8))

        overload_indices = {load['module_index'] for load in joint_torques if load['overloaded']}
        for idx, joint in enumerate(joints[:-1]):
            mesh = GeometryEngine.get_sphere_mesh(joint, 0.2)
            color = 'red' if idx in overload_indices else 'gold'
            self.surfaces.append(self.ax.plot_surface(*mesh, color=color, shade=True))

        end_pt = joints[-1]
        mesh_end = GeometryEngine.get_sphere_mesh(end_pt, 0.25)
        self.surfaces.append(self.ax.plot_surface(*mesh_end, color='lime', shade=True))

        if not self.trajectory or np.linalg.norm(self.trajectory[-1] - end_pt) > 0.05:
            self.trajectory.append(end_pt.copy())
        if len(self.trajectory) > 1:
            t_pts = np.array(self.trajectory)
            if self.traj_line:
                self.traj_line.set_data_3d(t_pts[:, 0], t_pts[:, 1], t_pts[:, 2])
            else:
                self.traj_line, = self.ax.plot(t_pts[:, 0], t_pts[:, 1], t_pts[:, 2], 'k--', lw=1, alpha=0.3)

        for load in joint_torques:
            text = f"{load['type']}{load['module_index'] + 1} |M|={load['torque_magnitude']:.1f}Nm"
            if load['type'] == 'B':
                text += f"\n|Taxis|={load['axis_torque']:.1f}Nm"
            text += f"\nlimit={load['max_torque']:.1f}Nm"
            
            box_face = '#ffdddd' if load['overloaded'] else 'white'
            box_edge = '#cc0000' if load['overloaded'] else '#666666'
            text_color = '#aa0000' if load['overloaded'] else 'black'
            
            label = self.ax.text(
                load['label_pos'][0], load['label_pos'][1], load['label_pos'][2],
                text, fontsize=8, color=text_color, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=box_face, alpha=0.85, edgecolor=box_edge)
            )
            self.torque_texts.append(label)

        if self.target_marker_pos is not None:
            m_target = GeometryEngine.get_sphere_mesh(self.target_marker_pos, 0.15)
            self.surfaces.append(self.ax.plot_surface(*m_target, color='orange', alpha=0.9))
            if self.target_marker_dir is not None:
                p_dir = self.target_marker_pos + self.target_marker_dir * 1.5
                l_dir, = self.ax.plot([self.target_marker_pos[0], p_dir[0]], 
                                     [self.target_marker_pos[1], p_dir[1]], 
                                     [self.target_marker_pos[2], p_dir[2]], 
                                     color='cyan', linestyle='--', linewidth=2)
                self.axis_lines.append(l_dir)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if not event.key: return
        key = event.key.lower()
        if key == 'c':
            self.trajectory = []
            if self.traj_line: self.traj_line.remove(); self.traj_line = None
            self.render()
        elif key == 'r':
            self.arm.reset()
            self.trajectory = []
            self.render()