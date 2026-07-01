import numpy as np
from geometry_utils import GeometryEngine
class RobotArm:
    def __init__(self, config_str):
        self.modules = []
        self._parse_config(config_str)
        self.rad_A = 0.15 
        self.rad_B = 0.10 

    def _parse_config(self, config_str):
        for line in config_str.strip().split('\n'):
            p = line.strip().replace(',', ' ').split()
            if not p: continue
            m_type = p[0].upper()
            if m_type == 'A':
                # A n_dir L
                self.modules.append({
                    'type': 'A', 
                    'n1': np.array([float(p[1]), float(p[2]), float(p[3])]), 
                    'L': float(p[4])
                })
            elif m_type == 'B':
                # B n_axis n_zero_dir L
                n1 = np.array([float(p[1]), float(p[2]), float(p[3])]) # 旋转轴
                n2 = np.array([float(p[4]), float(p[5]), float(p[6])]) # 零位杆件朝向
                L = float(p[7])
                # 确保 n2 垂直于 n1
                n1 /= np.linalg.norm(n1)
                n2 -= np.dot(n2, n1) * n1
                n2 /= np.linalg.norm(n2)
                self.modules.append({'type': 'B', 'n1': n1, 'n2': n2, 'L': L})

    def forward_kinematics(self, states):
        p, R_accum = np.array([0., 0., 0.]), np.eye(3)
        segments, joints, servo_frames = [], [p.copy()], []
        
        for i, m in enumerate(self.modules):
            val = states[i]
            rad = self.rad_A if m['type'] == 'A' else self.rad_B
            
            if m['type'] == 'A':
                d_glob = R_accum @ (m['n1'] / np.linalg.norm(m['n1']))
                seg_end = p + d_glob * m['L']
                segments.append({'type': 'A', 'start': p.copy(), 'end': seg_end.copy(), 'radius': rad})
                p = p + d_glob * val
            else:
                n1_glob = R_accum @ m['n1']
                n2_glob = R_accum @ m['n2']
                servo_frames.append({'pos': p.copy(), 'axis': n1_glob, 'zero': n2_glob})
                
                R_local = GeometryEngine.rodrigues_rotation(m['n1'], val)
                R_accum = R_accum @ R_local
                v_rod = R_accum @ m['n2']
                seg_end = p + v_rod * m['L']
                segments.append({'type': 'B', 'start': p.copy(), 'end': seg_end.copy(), 'radius': rad})
                p = seg_end.copy()
            
            joints.append(p.copy())
        return segments, joints, servo_frames


    def check_collision(self, curr_s, last_s, steps=5):
        for alpha in np.linspace(0, 1, steps):
            interp_s = [l + alpha * (c - l) for l, c in zip(last_s, curr_s)]
            segs, _, _ = self.forward_kinematics(interp_s)
            for i in range(len(segs)):
                for j in range(i + 2, len(segs)):
                    d = GeometryEngine.dist_segment_to_segment(segs[i]['start'], segs[i]['end'], segs[j]['start'], segs[j]['end'])
                    if d < (segs[i]['radius'] + segs[j]['radius']): return True
        return False