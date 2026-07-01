import time
import numpy as np
from robot_model import RobotArm
from visual import ArmVisualizer
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    # A [方向n] [行程L]
    # B [旋转轴n1] [零位朝向n2] [杆长L]
    config = """
    A 1 0 0 5
    A 0 1 0 5
    A 0 0 1 5
    B 0 0 1  1 0 0  4
    B 0 1 0  1 0 0  3
    B 0 1 0  1 0 0  2
    """
    
    robot = RobotArm(config)
    viz = ArmVisualizer(robot)
    
    
    # 可通过该方法调试算法
    plt.ion() 
    
    t = 0.0
    while plt.fignum_exists(viz.fig.number):
        # None以供手动调整滑条
        s1 = None
        s2 = None
        s3 = None
        theta1 = math.pi * math.sin(t)
        theta2 = 0.5 * math.cos(t * 2)

        theta3 = None
        
        
        viz.set_joint_states([s1, s2, s3, theta1, theta2, theta3])
        
        plt.pause(0.05) 
        t += 0.05