import sys
import numpy as np
import cv2
from config import OPENNI2_REDIST_PATH, IMG_WIDTH, IMG_HEIGHT, IS_WINDOWS
import math

# Windows 专用依赖：OpenNI2 (Astra/Astra Pro) 与 pyorbbecsdk (Orbbec Gemini)
# 在 macOS / Linux 上不强制安装，只有真用到对应相机时才会提示。
if IS_WINDOWS:
    try:
        from openni import openni2
        from openni import _openni2 as c_api
    except Exception as _e:
        openni2 = None
        c_api = None
        _OPENNI_IMPORT_ERROR = _e
    try:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, OBPropertyID
        from pyorbbecsdk import Context, TemporalFilter, HoleFillingFilter
    except Exception as _e:
        Pipeline = Config = OBSensorType = OBFormat = OBAlignMode = OBPropertyID = None
        Context = TemporalFilter = HoleFillingFilter = None
        _ORBBEC_IMPORT_ERROR = _e
else:
    openni2 = None
    c_api = None
    Pipeline = Config = OBSensorType = OBFormat = OBAlignMode = OBPropertyID = None
    Context = TemporalFilter = HoleFillingFilter = None

import config

class AstraCamera:
    def __init__(self):
        if not IS_WINDOWS or openni2 is None:
            raise RuntimeError(
                "AstraCamera 仅支持 Windows。当前平台: "
                f"{sys.platform}。请在 Mac 上使用 MacCamera / RealSense / VirtualCamera。"
            )
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
        if not IS_WINDOWS or Pipeline is None:
            raise RuntimeError(
                "GeminiCamera 依赖 pyorbbecsdk（Orbbec 官方 SDK），目前只提供 Windows 版本。"
                f"当前平台: {sys.platform}。请在 Mac 上使用 MacCamera / RealSense / VirtualCamera。"
            )
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


# =====================================================================
# 以下为 macOS / Linux 友好的相机后端
# 接口与 Windows 版本完全一致：get_intrinsics() / get_frames() / release()
# =====================================================================

class MacRGBCamera:
    """macOS / Linux 上的普通 RGB 摄像头（无深度）。

    - 优先使用 AVFoundation 后端 (macOS)
    - 失败回退到默认后端
    - 深度图填充为 0（毫米），几何分割会被关闭，pipeline 退化为纯 2D 检测
    """
    def __init__(self, index=0, width=640, height=480):
        if sys.platform == "darwin":
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            # 回退默认后端
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 index={index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

        # 写死一组合理的内参，方便 2D 流程
        self.fx, self.fy = 575.0, 575.0
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        print(f"MacRGBCamera 已打开 {self.width}x{self.height}（无深度，几何分割将不可用）")

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def get_frames(self):
        ret, bgr_img = self.cap.read()
        if not ret or bgr_img is None:
            return None, None, None
        # 构造空深度图（mm）
        d_arr = np.zeros((bgr_img.shape[0], bgr_img.shape[1]), dtype=np.float32)
        c_arr = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return bgr_img, c_arr, d_arr

    def release(self):
        self.cap.release()


class RealSenseCamera:
    """Intel RealSense D400 系列 (D435 / D455 ...) 在 macOS 上的后端。

    需要安装 pyrealsense2：
        pip install pyrealsense2

    若库未安装，给出明确报错。
    """
    def __init__(self, width=640, height=480, fps=30):
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise RuntimeError(
                "RealSenseCamera 需要 pyrealsense2，请在 Mac 上先 `pip install pyrealsense2`。\n"
                f"原始错误: {e}"
            )
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.width, self.height = intr.width, intr.height
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.cx, intr.cy

        # 同步到全局 config，便于 renderer 使用
        config.FX, config.FY = self.fx, self.fy
        config.CX, config.CY = self.cx, self.cy
        config.IMG_WIDTH, config.IMG_HEIGHT = self.width, self.height
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        config.U, config.V = u.flatten(), v.flatten()

        self.align = rs.align(rs.stream.color)
        print(f"RealSenseCamera 已打开 {self.width}x{self.height}")

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None

        bgr_img = np.asanyarray(color_frame.get_data())
        d_raw = np.asanyarray(depth_frame.get_data())
        d_arr = d_raw.astype(np.float32)  # RealSense depth 已经是 mm
        c_arr = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return bgr_img, c_arr, d_arr

    def release(self):
        self.pipeline.stop()