import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 路径配置 =================
OPENNI2_REDIST_PATH = os.path.join(
    BASE_DIR, 
    "OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows", 
    "Win64-Release", 
    "sdk", 
    "libs"
)

# 内参
FX, FY = 575.0, 575.0
CX, CY = 320.0, 240.0
IMG_WIDTH, IMG_HEIGHT = 640, 480

U, V = np.meshgrid(np.arange(IMG_WIDTH), np.arange(IMG_HEIGHT))
U, V = U.flatten(), V.flatten()

# ================= Camera =================
CAMERA_MOUSE_SENSITIVITY = 0.0012    
CAMERA_MOVE_SPEED = 0.02             

# ================= 渲染配置 =================
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
APPLE_CENTER_RADIUS = 0.005          # 球心半径
APPLE_SHELL_RESOLUTION = 12          # 球的网格分辨率

# =================  Tracker 配置 =================
MAX_APPLES = 5                       # 追踪目标数
TRACKER_DIST_THRESH = 0.03           # 匹配距离阈值
TRACKER_EMA_ALPHA = 0.6              # 平滑系数 (0~1, 越大越相信新观测值)

# ================= 图像处理过滤配置 =================
DEPTH_Z_MIN = 0.1                    # 有效深度下限
DEPTH_Z_MAX = 2.5                    # 有效深度上限

UI_DEF_BLUR_R = 5
GAUSSIAN_KERNEL = (UI_DEF_BLUR_R, UI_DEF_BLUR_R)             # 高斯滤波核大小
MORPH_OPEN_KERNEL = (3, 3)           # 开运算核大小 (去噪点)
MORPH_CLOSE_KERNEL = (5, 5)          # 闭运算核大小 (填补空洞)
MIN_CONTOUR_AREA = 300               # 最小有效轮廓面积 (像素)


UI_DEF_HUE_TOL = 12
UI_DEF_SAT_MIN = 100
UI_DEF_VAL_MIN = 50
UI_DEF_TIME_WIN = 3
UI_DEF_NORM_ANGLE = 30
UI_DEF_MIN_RAD = 15 
UI_DEF_MAX_RAD = 60
UI_DEF_CONFIRM_FRM = 4
UI_DEF_LOST_FRM = 10

# =================  点云与几何算法 =================
MIN_APPLE_RADIUS_M = 0.015           # 限制目标的最小半径 1.5cm
VOXEL_SIZE = 0.005                   # 点云下采样体素大小
NORMAL_SEARCH_RADIUS = 0.03          # 计算法线时所用的 KDTree 搜索半径
NORMAL_MAX_NN = 30                   # 法线估计最大邻居数
KDTREE_SEARCH_RADIUS = 0.012         # 发现过滤时用的 KDTree 搜索半径
DBSCAN_EPS = 0.015                   # DBSCAN 聚类距离
DBSCAN_MIN_POINTS = 15               # DBSCAN 聚类最少点数
SPHERE_FIT_MIN_PTS = 15              # 触发球面拟合的最少点数
PASSING_NEIGHBORS = 7                # 8个临近点中，通过法线连续性检查的点数