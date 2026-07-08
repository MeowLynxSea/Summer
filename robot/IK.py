import numpy as np
from robot_model import RobotArm

class RobotIKSolver:
    def __init__(self, robot: RobotArm, damp=0.01, step_size=0.3):
        self.robot = robot
        self.damp = damp           # 阻尼，防止奇异点
        self.step_size = step_size # 迭代步长
        
        self.v_limits = np.array([m['speed'] for m in robot.modules])

        q_min = []
        q_max = []
        for m in robot.modules:
            if m['type'] == 'A':
                q_min.append(0.0)
                q_max.append(m['L'])
            else:
                q_min.append(-np.pi) 
                q_max.append(np.pi)

        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)

        self.inv_W = self.v_limits ** 2

    def solve(self, target_pos, target_n, q_init, max_iter=100, tol=1e-4):
        """
        target_pos: np.array([x, y, z])
        target_n: np.array([nx, ny, nz]) 抓取指向
        q_init: 初始构型
        """
        q = np.array(q_init, dtype=float).copy()
        target_n = target_n / np.linalg.norm(target_n)
        
        for i in range(max_iter):
            p_ee, z_ee, p_list, z_list = self.robot.get_ik_data(q)
            
            err_p = target_pos - p_ee
            err_o = np.cross(z_ee, target_n)
            
            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_o) < tol:
                print(f"IK 收敛于第 {i} 次迭代")
                break
            # if(i%20 == 0):
            #     print(f"IK 第 {i} 次迭代: {np.linalg.norm(err_p)}, {np.linalg.norm(err_o)}")

            e = np.concatenate([err_p, err_o])
            
            J = self.robot.get_jacobian(p_ee, p_list, z_list)
            
            # 加权阻尼伪逆
            # A = J * inv_W * J.T + damp^2 * I
            A = (J * self.inv_W) @ J.T + (self.damp**2) * np.eye(6)
            try:
                lambda_vec = np.linalg.solve(A, e)
            except np.linalg.LinAlgError:
                print(f"IK 遇到奇异点，第 {i} 次迭代")
                break
            
            dq_task = self.inv_W * (J.T @ lambda_vec)
            
            # 零空间投影：关节中心化
            # 目标：使 q 靠近行程中间
            q_mid = (self.q_min + self.q_max) / 2.0
            dq_0 = (q_mid - q) * 0.1 
            
            # 零空间投影 dq_null = (I - J_inv_w * J) * dq_0
            # dq_0 也考虑 inv_W 权重
            weighted_dq0 = self.inv_W * dq_0
            dq_null = weighted_dq0 - self.inv_W * (J.T @ np.linalg.solve(A, J @ weighted_dq0))
            
            q += self.step_size * (dq_task + dq_null)
            q = np.clip(q, self.q_min, self.q_max)
        
        print("未收敛")
        return q