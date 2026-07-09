import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import cv2
from config import OPENNI2_REDIST_PATH, IMG_WIDTH, IMG_HEIGHT
import math
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, OBPropertyID
from pyorbbecsdk import Context, TemporalFilter, HoleFillingFilter
import config

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


class GeminiCamera:
    def __init__(self):
        print("正在初始化 Gemini 305")
        try:
            self.pipeline = Pipeline()
            self.config_obj = Config()

            color_profile = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
            depth_profile = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()

            self.width = color_profile.get_width()
            self.height = color_profile.get_height()

            self.config_obj.enable_stream(color_profile)
            self.config_obj.enable_stream(depth_profile)
            self.config_obj.set_align_mode(OBAlignMode.HW_MODE) 

            self.pipeline.start(self.config_obj)

            param = self.pipeline.get_camera_param()
            
            config.FX = param.rgb_intrinsic.fx
            config.FY = param.rgb_intrinsic.fy
            config.CX = param.rgb_intrinsic.cx
            config.CY = param.rgb_intrinsic.cy
            
            config.IMG_WIDTH = self.width
            config.IMG_HEIGHT = self.height

            print(f"重新生成投影网格以适配 {self.width}x{self.height}...")
            u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
            config.U = u.flatten()
            config.V = v.flatten()

            print(f"系统参数已更新: FX={config.FX:.2f}, CX={config.CX:.2f}")

        except Exception as e:
            print(f"Gemini 305 初始化失败: {e}")

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy
    
    def get_frames(self):
        frames = self.pipeline.wait_for_frames(100)
        if not frames: return None, None, None
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame: return None, None, None
     
        scale = depth_frame.get_depth_scale()

        d_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).copy()
        d_arr = d_raw.reshape((self.height, self.width)).astype(np.float32)*scale
        

        c_raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8).copy()
        c_yuv = c_raw.reshape((self.height, self.width, 2))
        bgr_img = cv2.cvtColor(c_yuv, cv2.COLOR_YUV2BGR_YUYV)
        c_arr = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        return bgr_img, c_arr, d_arr

    def release(self):
        self.pipeline.stop()