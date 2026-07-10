import time
from robot_model import RobotArm
from visual import ArmVisualizer
from IK import RobotIKSolver
from trajectory import EndEffectorPath, TrajectoryFollower
import numpy as np
import matplotlib.pyplot as plt


def build_demo_trajectory(start_pos, start_dir):
    start_pos = np.asarray(start_pos, dtype=float)
    start_dir = np.asarray(start_dir, dtype=float)

    return EndEffectorPath.from_waypoints(
        [
            (start_pos, start_dir),
            (np.array([8.2, 0.8, 1.2]), np.array([-1.0, 0.1, 0.2])),
            (np.array([6.0, 1.8, 2.8]), np.array([-1.0, 0.2, 0.4])),
            (np.array([3.8, 2.8, 4.3]), np.array([-0.8, 0.5, 0.6])),
            (np.array([1.0, 3.2, 5.0]), np.array([0.0, 1.0, 1.0])),
        ],
        min_duration=0.25,
        linear_speed=1.2,
        angular_speed=1.5,
    )

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

    follower = TrajectoryFollower()
    start_pos, start_dir = robot.get_ee_pose()
    follower.set_trajectory(build_demo_trajectory(start_pos, start_dir))
    
    plt.ion()

    t = 0.0
    last_wall_time = time.perf_counter()
    while plt.fignum_exists(viz.fig.number):
        now_wall_time = time.perf_counter()
        dt = min(max(now_wall_time - last_wall_time, 1e-4), 0.2)
        last_wall_time = now_wall_time
        t += dt
        
        current_q = robot.get_current_q()
        ee_pos, ee_dir = robot.get_ee_pose()
        target_pos, target_dir = follower.update_target(
            ee_pos,
            ee_dir,
            pos_tol=0.12,
            dir_tol=0.12,
        )
        
        q_cmd = solver.solve(target_pos, target_dir, current_q, max_iter=40, tol=1e-3)

        robot.set_target(q_cmd)
        status = robot.update(dt)
        if status == "COLLISION":
            print(f"检测到碰撞！t={t:.2f}")
        viz.set_target_visual(target_pos, target_dir)
        viz.render(t)
        plt.pause(0.001)
