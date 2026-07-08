import time
from robot_model import RobotArm
from visual import ArmVisualizer
from IK import RobotIKSolver
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
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
    solver = RobotIKSolver(robot)
    viz = ArmVisualizer(robot)

    apple_pos = np.array([1.0, 3.2, 5.0])
    grab_dir = np.array([0.0, 1.0, 1.0])

    current_q = np.zeros(len(robot.modules))
    target_q = solver.solve(apple_pos, grab_dir, current_q)
    
    plt.ion()

    t = 0.0
    last_wall_time = time.perf_counter()
    while plt.fignum_exists(viz.fig.number):
        now_wall_time = time.perf_counter()
        dt = min(max(now_wall_time - last_wall_time, 1e-4), 0.2)
        last_wall_time = now_wall_time
        t += dt
        
        if not robot.collision_locked:
            # apple_pos += np.array([0.0, 0.0, 0.1]) * dt
            grab_dir = np.array([np.cos(0.5*t), np.sin(0.5*t), 1.0])
        # # None以供手动调整滑条
        # s1 = None
        # s2 = None
        # s3 = None
        # theta1 = math.pi * math.sin(t)
        # # theta2 = 0.5 * math.cos(t * 2)

        # theta3 = None

        pee,zee = robot.get_ee_pose()
        if (np.linalg.norm(pee - apple_pos) > 0.1 or
                np.linalg.norm(zee- grab_dir) > 0.1):
            print("new apple")
            current_q = robot.get_current_q()
            target_q = solver.solve(apple_pos, grab_dir, current_q)
        
        
        robot.set_target(target_q) 
        status = robot.update(dt)
        if status == "COLLISION":
            print(f"检测到碰撞！t={t:.2f}")

        viz.set_target_visual(apple_pos, grab_dir)
        viz.render(t)
        plt.pause(0.001)
