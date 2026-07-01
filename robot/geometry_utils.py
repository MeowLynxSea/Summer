import numpy as np
class GeometryEngine:
    @staticmethod
    def rodrigues_rotation(axis, theta):
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-9: return np.eye(3)
        axis /= norm
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    @staticmethod
    def dist_segment_to_segment(p1, p2, q1, q2):
        d1, d2, r = p2 - p1, q2 - q1, p1 - q1
        a, b, c = np.dot(d1, d1), np.dot(d1, d2), np.dot(d2, d2)
        d, e = np.dot(d1, r), np.dot(d2, r)
        det = a * c - b * b
        s = np.clip((b * e - c * d) / det, 0, 1) if det > 1e-6 else 0.0
        t = np.clip((s * b + e) / c, 0, 1) if c > 1e-6 else 0.0
        s = np.clip((t * b - d) / a, 0, 1) if a > 1e-6 else 0.0
        return np.linalg.norm((p1 + s * d1) - (q1 + t * d2))

    @staticmethod
    def get_cylinder_mesh(start, end, radius, res=6):
        v = end - start
        L = np.linalg.norm(v)
        if L < 1e-6: return None
        v_n = v / L
        z = np.linspace(0, L, 2); theta = np.linspace(0, 2*np.pi, res)
        z_g, th_g = np.meshgrid(z, theta)
        x_g, y_g = radius * np.cos(th_g), radius * np.sin(th_g)
        z_axis = np.array([0, 0, 1]); rot_axis = np.cross(z_axis, v_n)
        if np.linalg.norm(rot_axis) < 1e-6:
            R = np.eye(3) if v_n[2] > 0 else -np.eye(3)
        else:
            angle = np.arccos(np.clip(np.dot(z_axis, v_n), -1, 1))
            R = GeometryEngine.rodrigues_rotation(rot_axis, angle)
        pts = np.stack([x_g.flatten(), y_g.flatten(), z_g.flatten()])
        pts_r = R @ pts
        return (pts_r[0,:].reshape(x_g.shape) + start[0],
                pts_r[1,:].reshape(x_g.shape) + start[1],
                pts_r[2,:].reshape(x_g.shape) + start[2])

    @staticmethod
    def get_sphere_mesh(center, radius, res=6):
        u, v = np.mgrid[0:2*np.pi:res*1j, 0:np.pi:res*1j]
        return (radius * np.cos(u) * np.sin(v) + center[0],
                radius * np.sin(u) * np.sin(v) + center[1],
                radius * np.cos(v) + center[2])