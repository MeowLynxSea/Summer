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

    class _Cam:
        def __init__(self, cap):
            self.cap = cap
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.IMG_WIDTH
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.IMG_HEIGHT
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

    print(f"[Live] 使用摄像头 {cam_width}x{cam_height}（无深度）".replace(
        "cam_width", str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    ).replace("cam_height", str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
    return _Cam(cap)


def main():
    cam = _open_camera()

    # 把录像真实分辨率/内参同步到全局 config，供 vision / renderer 使用
    if hasattr(cam, "width"):
        config.IMG_WIDTH = cam.width
        config.IMG_HEIGHT = cam.height
    if hasattr(cam, "fx"):
        config.FX, config.FY = cam.fx, cam.fy
        config.CX, config.CY = cam.cx, cam.cy
    import numpy as np
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

    print("回放/演示模式启动；ESC 退出。")
    last_time = time.perf_counter()
    try:
        while fps_camera.running:
            now = time.perf_counter()
            dt = max(0.001, min(now - last_time, 0.5))
            last_time = now

            bgr_img, c_arr, d_arr = cam.get_frames()
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

            if cv2.waitKey(1) == 27:
                break
    finally:
        cam.release()
        renderer.release()


if __name__ == "__main__":
    main()