import os
import cv2
import numpy as np
from datetime import datetime
import time
class DataRecorder:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.is_recording = False
        self.video_out_rgb = None
        self.depth_bin_file = None
        
        for folder in ['videos', 'snapshots']:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def start_recording(self):
        if not self.is_recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            rgb_path = f'videos/video_{timestamp}_rgb.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            self.video_out_rgb = cv2.VideoWriter(rgb_path, fourcc, 20.0, (self.width, self.height))
            
            # 深度二进制流
            self.depth_bin_path = f'videos/video_{timestamp}_depth.bin'
            self.depth_bin_file = open(self.depth_bin_path, 'wb')
            
            self.is_recording = True
            print(f"开始录制: {timestamp}")

    def record_frame(self, bgr_img, d_arr):
        if self.is_recording:
            self.video_out_rgb.write(bgr_img)
            # 原始 16位 深度数据
            self.depth_bin_file.write(d_arr.astype(np.uint16).tobytes())

    def stop_recording(self):
        if self.is_recording:
            if self.video_out_rgb: self.video_out_rgb.release()
            if self.depth_bin_file: self.depth_bin_file.close()
            self.is_recording = False
            print("停止录制，数据已保存。")
    def save_snapshot(self, bgr_img, d_arr):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
        cv2.imwrite(f'snapshots/snap_{timestamp}_rgb.png', bgr_img)
        np.save(f'snapshots/snap_{timestamp}_depth.npy', d_arr.astype(np.uint16))
        print(f"截取: snap_{timestamp}")