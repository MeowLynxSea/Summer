import numpy as np
from geometry_utils import GeometryEngine
class RobotArm:
    def __init__(self, config_str):
        self.modules = []
        self._parse_config(config_str)
        self.rad_A = 0.15 
        self.rad_B = 0.10 
        self.gravity = np.array([0.0, 0.0, -9.81])

    def _parse_config(self, config_str):
        for line in config_str.strip().split('\n'):
            p = line.strip().replace(',', ' ').split()
            if not p:
                continue
            m_type = p[0].upper()
            if m_type == 'A':
                # A n_dir L rod_mass module_mass max_torque speed
                self.modules.append({
                    'type': 'A', 
                    'n1': np.array([float(p[1]), float(p[2]), float(p[3])]), 
                    'L': float(p[4]),
                    'rod_mass': float(p[5]),
                    'module_mass': float(p[6]),
                    'max_torque': float(p[7]),
                    'speed': float(p[8]),
                })
            elif m_type == 'B':
                # B n_axis n_zero_dir L rod_mass module_mass max_torque speed
                n1 = np.array([float(p[1]), float(p[2]), float(p[3])])  # 旋转轴
                n2 = np.array([float(p[4]), float(p[5]), float(p[6])])  # 零位杆件朝向
                L = float(p[7])
                rod_mass = float(p[8])
                module_mass = float(p[9])
                max_torque = float(p[10])
                speed = float(p[11])
                # 确保 n2 垂直于 n1
                n1 /= np.linalg.norm(n1)
                n2 -= np.dot(n2, n1) * n1
                n2 /= np.linalg.norm(n2)
                self.modules.append({
                    'type': 'B',
                    'n1': n1,
                    'n2': n2,
                    'L': L,
                    'rod_mass': rod_mass,
                    'module_mass': module_mass,
                    'max_torque': max_torque,
                    'speed': speed,
                })

    @staticmethod
    def _orthogonal_unit_vector(vec):
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        vec = vec / norm
        basis = np.array([1.0, 0.0, 0.0]) if abs(vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ortho = np.cross(vec, basis)
        ortho_norm = np.linalg.norm(ortho)
        if ortho_norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return ortho / ortho_norm

    def _compute_joint_torques(self, node_infos, mass_points):
        joint_torques = []
        for node in node_infos:
            torque = np.zeros(3, dtype=float)
            for mass_pt in mass_points:
                if mass_pt['module_index'] < node['module_index']:
                    continue
                gravity_force = mass_pt['mass'] * self.gravity
                arm = mass_pt['pos'] - node['pos']
                torque += np.cross(arm, gravity_force)

            axis = node['axis'] / np.linalg.norm(node['axis'])
            axis_torque_signed = float(np.dot(torque, axis))
            bending = torque - axis_torque_signed * axis
            limit_torque = node['max_torque']
            load_torque = float(abs(axis_torque_signed)) if node['type'] == 'B' else float(np.linalg.norm(torque))
            joint_torques.append({
                'module_index': node['module_index'],
                'type': node['type'],
                'pos': node['pos'].copy(),
                'axis': axis.copy(),
                'label_pos': node['pos'] + node['label_dir'] * 0.45,
                'torque_vector': torque,
                'torque_magnitude': float(np.linalg.norm(torque)),
                'axis_torque_signed': axis_torque_signed,
                'axis_torque': float(abs(axis_torque_signed)),
                'bending_magnitude': float(np.linalg.norm(bending)),
                'max_torque': limit_torque,
                'load_torque': load_torque,
                'overloaded': load_torque > limit_torque if limit_torque > 0 else False,
            })
        return joint_torques

    def forward_kinematics(self, states):
        p, R_accum = np.array([0., 0., 0.]), np.eye(3)
        segments, joints, servo_frames = [], [p.copy()], []
        node_infos, mass_points = [], []
        
        for i, m in enumerate(self.modules):
            val = states[i]
            rad = self.rad_A if m['type'] == 'A' else self.rad_B
            
            if m['type'] == 'A':
                d_glob = R_accum @ (m['n1'] / np.linalg.norm(m['n1']))
                axis_hint = self._orthogonal_unit_vector(d_glob)
                servo_frames.append({'pos': p.copy(), 'axis': d_glob, 'zero': axis_hint, 'type': 'A'})
                node_infos.append({
                    'module_index': i,
                    'type': 'A',
                    'pos': p.copy(),
                    'axis': d_glob.copy(),
                    'label_dir': axis_hint.copy(),
                    'max_torque': m['max_torque'],
                })
                seg_end = p + d_glob * m['L']
                segments.append({'type': 'A', 'start': p.copy(), 'end': seg_end.copy(), 'radius': rad})
                active_end = p + d_glob * val
                if m['rod_mass'] > 0 and val > 1e-9:
                    mass_points.append({
                        'module_index': i,
                        'mass': m['rod_mass'],
                        'pos': p + d_glob * (val * 0.5),
                    })
                if m['module_mass'] > 0:
                    mass_points.append({
                        'module_index': i,
                        'mass': m['module_mass'],
                        'pos': active_end.copy(),
                    })
                p = active_end
            else:
                n1_glob = R_accum @ m['n1']
                n2_glob = R_accum @ m['n2']
                servo_frames.append({'pos': p.copy(), 'axis': n1_glob, 'zero': n2_glob, 'type': 'B'})
                node_infos.append({
                    'module_index': i,
                    'type': 'B',
                    'pos': p.copy(),
                    'axis': n1_glob.copy(),
                    'label_dir': n2_glob.copy(),
                    'max_torque': m['max_torque'],
                })
                
                R_local = GeometryEngine.rodrigues_rotation(m['n1'], val)
                R_accum = R_accum @ R_local
                v_rod = R_accum @ m['n2']
                seg_end = p + v_rod * m['L']
                segments.append({'type': 'B', 'start': p.copy(), 'end': seg_end.copy(), 'radius': rad})
                if m['rod_mass'] > 0:
                    mass_points.append({
                        'module_index': i,
                        'mass': m['rod_mass'],
                        'pos': p + v_rod * (m['L'] * 0.5),
                    })
                if m['module_mass'] > 0:
                    mass_points.append({
                        'module_index': i,
                        'mass': m['module_mass'],
                        'pos': seg_end.copy(),
                    })
                p = seg_end.copy()
            
            joints.append(p.copy())
        joint_torques = self._compute_joint_torques(node_infos, mass_points)
        return segments, joints, servo_frames, joint_torques


    def check_collision(self, curr_s, last_s, steps=5):
        for alpha in np.linspace(0, 1, steps):
            interp_s = [l + alpha * (c - l) for l, c in zip(last_s, curr_s)]
            segs, _, _, _ = self.forward_kinematics(interp_s)
            for i in range(len(segs)):
                for j in range(i + 2, len(segs)):
                    d = GeometryEngine.dist_segment_to_segment(segs[i]['start'], segs[i]['end'], segs[j]['start'], segs[j]['end'])
                    if d < (segs[i]['radius'] + segs[j]['radius']):
                        return True
        return False
