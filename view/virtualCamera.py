import cv2
import numpy as np


class VirtualCamera:
    """回放 recorder.py 录制的 RGB .avi + 深度 .bin 数据。

    自动从 .avi 拿真实分辨率，再从 .bin 大小 / 视频帧数反算深度帧字节数，
    支持非 640x480 的录制（Orbbec Gemini 305 默认 848x530 等）。
    """

    def __init__(self, video_path, bin_path, width=None, height=None):
        print(f"加载: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        self.bin_file = open(bin_path, 'rb')

        vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vframes = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        # .bin 总字节数 / 视频帧数 = 单帧深度字节数
        self.bin_file.seek(0, 2)
        bin_size = self.bin_file.tell()
        self.bin_file.seek(0)
        frame_bytes = bin_size // vframes
        pixels_per_frame = frame_bytes // 2  # uint16

        # 反推 depth 宽高（默认正方形；多数相机 RGB/Depth 同分辨率）
        depth_w = int(pixels_per_frame ** 0.5)
        depth_h = pixels_per_frame // depth_w
        if depth_w * depth_h != pixels_per_frame:
            # 兜底：用视频分辨率
            depth_w, depth_h = vw, vh

        # 优先用视频分辨率，深度若与视频不同则视为错配
        self.width = vw if width is None else width
        self.height = vh if height is None else height
        if depth_w * depth_h != self.width * self.height:
            self.width, self.height = depth_w, depth_h
        self.frame_size = self.width * self.height * 2

        # 默认内参
        self.fx, self.fy = 580.0, 580.0
        self.cx, self.cy = self.width / 2.0, self.height / 2.0

        print(f"  RGB={vw}x{vh}, Depth={depth_w}x{depth_h}, "
              f"frames={vframes}, frame_bytes={frame_bytes}")

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def get_total_frames(self):
        return max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def get_current_frame_index(self):
        # POS_FRAMES 是即将被读取的下一帧索引；最近一次显示的帧 = idx - 1
        # 当 POS_FRAMES == 0（刚 seek 到开头）时，认为当前就是第 0 帧
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return max(0, pos - 1) if pos > 0 else 0

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def seek(self, frame_idx):
        """跳到指定帧（0-based）。同步 .bin 文件指针。"""
        total = self.get_total_frames()
        frame_idx = max(0, min(int(frame_idx), total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.bin_file.seek(frame_idx * self.frame_size)

    def get_frames(self):
        ret, bgr_img = self.cap.read()
        depth_data = self.bin_file.read(self.frame_size)

        # 循环播放 / 数据耗尽
        if not ret or len(depth_data) != self.frame_size:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.bin_file.seek(0)
            ret, bgr_img = self.cap.read()
            depth_data = self.bin_file.read(self.frame_size)
            if not ret or len(depth_data) != self.frame_size:
                return None, None, None

        d_raw = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.height, self.width)
        d_arr = d_raw.astype(np.float32)
        c_arr = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return bgr_img, c_arr, d_arr

    def release(self):
        self.cap.release()
        self.bin_file.close()