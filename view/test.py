import cv2

def scan_cameras():
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera Index {i}", frame)
                print(f"找到摄像头索引: {i}")
            cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import multiprocessing
import psutil

def print_cpu_info():
    logical_cores = multiprocessing.cpu_count()
    # logical=False 获取的是物理核心（不含超线程）
    physical_cores = psutil.cpu_count(logical=False) 
    
    print("="*30)
    print(f"硬件核心检测:")
    print(f"物理核心数: {physical_cores}")
    print(f"逻辑核心数 (线程): {logical_cores}")
    print(f"当前算法可用核心: {logical_cores}")
    print("="*30)

print_cpu_info()