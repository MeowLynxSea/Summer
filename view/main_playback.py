# main_playback.py
# macOS / Linux 上零依赖入口：
#   - 默认从 videos/ 目录回放 recorder.py 录制的 .avi + .bin
#   - 若 videos/ 为空，自动用 OpenCV 打开 Mac 内置/USB 摄像头跑纯 2D 调参
#
# 用法：
#   cd view
#   python3 main_playback.py
import os
import sys
import time
import glob
import cv2
import numpy as np
import config


PLAYBACK_DIR = "videos"


def _find_latest_pair():
    """在 videos/ 里找最新的一对 (_rgb.avi, _depth.bin)。"""
    rgbs = sorted(glob.glob(os.path.join(PLAYBACK_DIR, "video_*_rgb.avi")))
    for rgb in reversed(rgbs):
        stem = rgb[:-len("_rgb.avi")]
        depth = stem + "_depth.bin"
        if os.path.exists(depth):
            return rgb, depth
    return None, None


def _open_camera():
    """优先回放录像；没有则回退到摄像头。"""
    rgb, depth = _find_latest_pair()
    if rgb and depth:
        from virtualCamera import VirtualCamera
        print(f"[Playback] 回放录像: {rgb}")
        return VirtualCamera(rgb, depth,
                              width=config.IMG_WIDTH, height=config.IMG_HEIGHT)

    cam_w = 0
    cam_h = 0
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("找不到任何摄像头，也无法回放录像（videos/ 为空）")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.IMG_HEIGHT)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.IMG_WIDTH
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.IMG_HEIGHT

    class _Cam:
        def __init__(self, cap, w, h):
            self.cap = cap
            self.width = w
            self.height = h
        def get_intrinsics(self):
            return 575.0, 575.0, self.width / 2.0, self.height / 2.0
        def get_frames(self):
            ret, bgr = self.cap.read()
            if not ret or bgr is None:
                return None, None, None
            d = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.float32)
            c = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return bgr, c, d
        def release(self):
            self.cap.release()

    print(f"[Live] 使用摄像头 {cam_w}x{cam_h}（无深度）")
    return _Cam(cap, cam_w, cam_h)


# ============== 视频模式进度条 UI ==============

class VideoController:
    """视频回放控制：暂停 / 进度条拖拽。仅在 VirtualCamera 模式下启用。

    注意：进度条独立渲染在 "Playback Timeline" 窗口，鼠标回调也挂在这个
    窗口上；hit-test 使用窗口实际尺寸，而非 cam 帧尺寸。
    """

    WIN_NAME = "Playback Timeline"
    WIN_HEIGHT = 90          # 时间线窗口高度
    BAR_HEIGHT = 14          # 进度条高度
    BAR_MARGIN_X = 30        # 进度条左右留白
    BAR_MARGIN_BOTTOM = 24   # 进度条距底
    BAR_HANDLE_W = 8         # 拖拽手柄宽度
    HIT_PAD_Y = 12           # 点击区垂直扩展
    HIT_PAD_X = 6            # 点击区水平扩展

    def __init__(self, cam):
        self.cam = cam
        self.paused = False
        self.dragging = False
        self.last_drawn_frame_idx = -1
        self.total = cam.get_total_frames()
        self.fps = cam.get_fps()

        # 创建独立时间线窗口（固定宽度 = 视频宽度），然后注册鼠标回调
        # 必须先 cv2.namedWindow，再 setMouseCallback 才能正确绑定
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_NAME, cam.width, self.WIN_HEIGHT)
        cv2.setMouseCallback(self.WIN_NAME, self._on_mouse)

    @property
    def win_size(self):
        """获取时间线窗口的实际宽高。"""
        try:
            x, y, w, h = cv2.getWindowImageRect(self.WIN_NAME)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        return self.cam.width, self.WIN_HEIGHT

    def _bar_rect(self, frame_w, frame_h):
        x1 = self.BAR_MARGIN_X
        x2 = frame_w - self.BAR_MARGIN_X
        y2 = frame_h - self.BAR_MARGIN_BOTTOM
        y1 = y2 - self.BAR_HEIGHT
        return x1, y1, x2, y2

    def _hit_test(self, x, y):
        w, h = self.win_size
        x1, y1, x2, y2 = self._bar_rect(w, h)
        return (x1 - self.HIT_PAD_X <= x <= x2 + self.HIT_PAD_X
                and y1 - self.HIT_PAD_Y <= y <= y2 + self.HIT_PAD_Y)

    def _x_to_frame(self, x):
        w, _ = self.win_size
        x1, _, x2, _ = self._bar_rect(w, 0)
        ratio = (x - x1) / max(1, (x2 - x1))
        ratio = max(0.0, min(1.0, ratio))
        return int(ratio * (self.total - 1))

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._hit_test(x, y):
                self.dragging = True
                self._seek_at(x)
                print(f"[Playback] 拖拽开始 -> frame {self.last_drawn_frame_idx}")
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self._seek_at(x)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                print(f"[Playback] 拖拽结束 -> frame {self.last_drawn_frame_idx}")

    def _seek_at(self, x):
        idx = self._x_to_frame(x)
        self.cam.seek(idx)
        self.last_drawn_frame_idx = idx

    def toggle_pause(self):
        self.paused = not self.paused
        print("[Playback] 已暂停" if self.paused else "[Playback] 继续播放")

    def draw(self, img):
        """在合成图上叠加进度条 + 时间码 + 暂停标识。"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self._bar_rect(w, h)

        # 当前帧（拖拽时直接用 last_drawn_frame_idx，避免与正在读取的帧错位）
        if self.dragging:
            cur = self.last_drawn_frame_idx
        else:
            cur = self.cam.get_current_frame_index()
            self.last_drawn_frame_idx = cur
        cur = max(0, min(cur, self.total - 1))
        ratio = cur / max(1, self.total - 1)

        # 整张面板深底
        img[:] = (30, 30, 30)

        # 半透明深色背景条
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # 已播放部分
        fx = int(x1 + ratio * (x2 - x1))
        cv2.rectangle(img, (x1, y1), (fx, y2), (0, 200, 255), -1)

        # 描边
        cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 1)

        # 拖拽手柄
        cv2.rectangle(
            img,
            (fx - self.BAR_HANDLE_W // 2, y1 - 3),
            (fx + self.BAR_HANDLE_W // 2, y2 + 3),
            (255, 255, 255),
            -1,
        )

        # 时间码
        def _fmt(n):
            s = n / self.fps if self.fps > 0 else 0
            m, ss = divmod(s, 60)
            return f"{int(m):02d}:{ss:05.2f}"

        time_text = f"{_fmt(cur)} / {_fmt(self.total - 1)}  [{cur + 1}/{self.total}]"
        cv2.putText(img, time_text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        # 暂停标识
        if self.paused:
            cv2.putText(img, " PAUSED", (x2 - 90, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cx, cy = w // 2, h // 2
            cv2.rectangle(img, (cx - 18, cy - 22), (cx - 6, cy + 22), (0, 0, 255), -1)
            cv2.rectangle(img, (cx + 6, cy - 22), (cx + 18, cy + 22), (0, 0, 255), -1)

        # 操作提示
        cv2.putText(
            img,
            "Space: Pause/Resume   |   Drag bar to seek   |   ESC: Quit",
            (x1, h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA,
        )


def main():
    cam = _open_camera()
    is_video_mode = isinstance(cam, __import__("virtualCamera").VirtualCamera)

    # 把录像真实分辨率/内参同步到全局 config，供 vision / renderer 使用
    if hasattr(cam, "width"):
        config.IMG_WIDTH = cam.width
        config.IMG_HEIGHT = cam.height
    if hasattr(cam, "fx"):
        config.FX, config.FY = cam.fx, cam.fy
        config.CX, config.CY = cam.cx, cam.cy
    u, v = np.meshgrid(np.arange(config.IMG_WIDTH), np.arange(config.IMG_HEIGHT))
    config.U, config.V = u.flatten(), v.flatten()
    print(f"config 已同步: {config.IMG_WIDTH}x{config.IMG_HEIGHT}, "
          f"FX={config.FX:.1f} FY={config.FY:.1f} "
          f"({config.CX:.1f},{config.CY:.1f})")

    from renderer import SceneRenderer
    from tracker import AppleTracker
    from vision import VisionProcessor
    from alg import segment_and_filter_apples
    from camera_3d import FPSCamera

    fps_camera = FPSCamera()
    tracker = AppleTracker()
    vision = VisionProcessor()
    renderer = SceneRenderer()

    video_ctrl = VideoController(cam) if is_video_mode else None

    hint = ("回放/演示模式启动；ESC 退出。"
            + (" [视频] 空格暂停，进度条可拖拽。" if is_video_mode else ""))
    print(hint)

    last_time = time.perf_counter()
    # 缓存上一帧，避免暂停时反复推进帧位置
    cached = {"bgr": None, "c": None, "d": None}

    def _ensure_frame_at(target_idx):
        """让 cam 内部指针停在 target_idx，并读取该帧用于显示/识别。"""
        cur = cam.get_current_frame_index()
        if cur != target_idx:
            cam.seek(target_idx)
        bgr, c, d = cam.get_frames()
        if bgr is None:
            return cached["bgr"], cached["c"], cached["d"]
        cached["bgr"], cached["c"], cached["d"] = bgr, c, d
        # seek 会前进 1 帧；如果连续两次同样的 target_idx 不会重复前进
        # 因为 get_current_frame_index 返回 seek 设置的位置（再 -1）
        # 实际行为：seek(idx) -> POS=idx -> get_frames() read -> POS=idx+1
        # 下次再进来 cur == idx+1 != target_idx(idx)，会再次 seek 回 idx
        # 不会持续前进。✓
        return bgr, c, d

    try:
        while fps_camera.running:
            now = time.perf_counter()
            dt = max(0.001, min(now - last_time, 0.5))
            last_time = now

            paused = video_ctrl.paused if video_ctrl else False
            dragging = (video_ctrl.dragging if video_ctrl else False)

            if paused:
                # 保持当前画面，不前进帧
                if cached["bgr"] is not None:
                    bgr_img, c_arr, d_arr = cached["bgr"], cached["c"], cached["d"]
                else:
                    bgr_img, c_arr, d_arr = cam.get_frames()
                    if bgr_img is not None:
                        cached["bgr"], cached["c"], cached["d"] = bgr_img, c_arr, d_arr
            elif dragging:
                # 拖拽中：停在拖拽目标帧
                target = video_ctrl.last_drawn_frame_idx
                bgr_img, c_arr, d_arr = _ensure_frame_at(target)
            else:
                bgr_img, c_arr, d_arr = cam.get_frames()
                if bgr_img is not None:
                    cached["bgr"], cached["c"], cached["d"] = bgr_img, c_arr, d_arr

            if bgr_img is None:
                continue

            acc_mask, target_pts_3d, params, d_arr_filtered = vision.process(bgr_img, d_arr)
            raw_apples = segment_and_filter_apples(
                target_pts_3d, params.norm_angle, params.min_rad, params.max_rad
            )
            confirmed_apples = tracker.update(raw_apples, params.confirm_f, params.lost_f, dt)

            if len(confirmed_apples) != fps_camera.last_apple_count:
                print(f"当前追踪 {len(confirmed_apples)} 个苹果")
                fps_camera.last_apple_count = len(confirmed_apples)

            fps_camera.update()

            renderer.update_3d_environment(c_arr, d_arr_filtered)
            renderer.update_apples(confirmed_apples)
            renderer.update_camera_view(fps_camera.get_extrinsic())
            renderer.show_2d_windows(bgr_img, d_arr_filtered, acc_mask, confirmed_apples)

            # 视频模式：独立一个时间线窗口，叠在 Segmentation Result 之下
            if video_ctrl is not None:
                _, win_h = video_ctrl.win_size
                timeline = np.zeros((win_h, cam.width, 3), dtype=np.uint8)
                video_ctrl.draw(timeline)
                cv2.imshow(video_ctrl.WIN_NAME, timeline)

            # 键盘
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32 and video_ctrl is not None:
                video_ctrl.toggle_pause()
    finally:
        cam.release()
        renderer.release()


if __name__ == "__main__":
    main()