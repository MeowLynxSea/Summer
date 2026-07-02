import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import cv2
from config import OPENNI2_REDIST_PATH, IMG_WIDTH, IMG_HEIGHT
import math


class AstraCamera:
    def __init__(self):
        print("初始化 OpenNI2")
        try:
            openni2.initialize(OPENNI2_REDIST_PATH)
            self.dev = openni2.Device.open_any()
            self.depth_stream = self.dev.create_depth_stream()
            self.color_stream = self.dev.create_color_stream()
            
            h_fov = self.depth_stream.get_horizontal_fov()
            v_fov = self.depth_stream.get_vertical_fov()

            vm = self.depth_stream.get_video_mode()
            self.width = vm.resolutionX
            self.height = vm.resolutionY

            # 计算内参
            self.fx = self.width / (2 * math.tan(h_fov / 2))
            self.fy = self.height / (2 * math.tan(v_fov / 2))
            self.cx = (self.width - 1) / 2.0
            self.cy = (self.height - 1) / 2.0

            self.dev.set_image_registration_mode(c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
            
            self.depth_stream.start()
            self.color_stream.start()
            print("初始化成功")
        except Exception as e:
            print(f"初始化失败: {e}")
            exit()

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy
    
    def get_frames(self):
        d_frame = self.depth_stream.read_frame()
        c_frame = self.color_stream.read_frame()
        
        d_raw = np.frombuffer(d_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(self.height, self.width)
        c_raw = np.frombuffer(c_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(self.height, self.width, 3)
    
        d_arr = d_raw.astype(np.float32).copy()
        c_arr = c_raw.copy()
        bgr_img = cv2.cvtColor(c_arr, cv2.COLOR_RGB2BGR)
    
        return bgr_img, c_arr, d_arr

    def release(self):
        self.depth_stream.stop()
        self.color_stream.stop()
        self.dev.close()
        openni2.unload()