import cv2
import numpy as np

class VirtualCamera:
    def __init__(self, video_path, bin_path, width=640, height=480):
        print(f"加载: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        self.bin_path = bin_path
        self.bin_file = open(bin_path, 'rb')
        self.width = width
        self.height = height
        self.frame_size = width * height * 2 # uint16 = 2 bytes
        
        # 默认内参
        self.fx, self.fy, self.cx, self.cy = (580.0, 580.0, width/2, height/2)

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def get_frames(self):
        ret, bgr_img = self.cap.read()
        
        # Depth 二进制块
        depth_data = self.bin_file.read(self.frame_size)
        
        # 循环播放
        if not ret or not depth_data:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.bin_file.seek(0)
            ret, bgr_img = self.cap.read()
            depth_data = self.bin_file.read(self.frame_size)

        d_raw = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.height, self.width)
        d_arr = d_raw.astype(np.float32) 
        
        c_arr = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        return bgr_img, c_arr, d_arr

    def release(self):
        self.cap.release()
        self.bin_file.close()