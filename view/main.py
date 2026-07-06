# main.py
import cv2
import numpy as np
import open3d as o3d
import config
from hardware import AstraCamera
from camera_3d import FPSCamera
from tracker import AppleTracker
from alg import segment_and_filter_apples
from vision import VisionProcessor
from renderer import SceneRenderer
from recorder import DataRecorder
from virtualCamera import VirtualCamera
import time
import os

def main():
    cam_hardware = AstraCamera()
    config.FX, config.FY, config.CX, config.CY = cam_hardware.get_intrinsics() #更新内参
    print("相机内参：", config.FX, config.FY, config.CX, config.CY)

    fps_camera = FPSCamera()
    tracker = AppleTracker()
    vision = VisionProcessor()   
    renderer = SceneRenderer() 

    print("TAB 键解锁/锁定鼠标，ESC 退出。")

    
    last_time = time.perf_counter()
    try:
        while fps_camera.running:

            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time
            dt = max(0.001, min(dt, 0.5))

            bgr_img, c_arr, d_arr = cam_hardware.get_frames()

            acc_mask, target_pts_3d, params,d_arr_filtered = vision.process(bgr_img, d_arr)

            raw_apples = segment_and_filter_apples(target_pts_3d, params.norm_angle, params.min_rad, params.max_rad)

            confirmed_apples = tracker.update(raw_apples, params.confirm_f, params.lost_f,dt)
            
            if len(confirmed_apples) != fps_camera.last_apple_count:
                print(f"当前追踪 {len(confirmed_apples)} 个苹果")
                fps_camera.last_apple_count = len(confirmed_apples)

            fps_camera.update()

            renderer.update_3d_environment(c_arr, d_arr_filtered)
            renderer.update_apples(confirmed_apples)
            renderer.update_camera_view(fps_camera.get_extrinsic())
            renderer.show_2d_windows(bgr_img, d_arr_filtered, acc_mask,confirmed_apples)

            if cv2.waitKey(1) == 27: 
                break
    finally:
        print("exit")
        fps_camera.release()
        cam_hardware.release()
        renderer.release()

def record():
    cam = AstraCamera()
    width, height = cam.width, cam.height
    
    recorder = DataRecorder(width, height)
    
    print("=== 苹果识别数据采集系统 ===")
    print(f"分辨率: {width}x{height}")
    print("  [R]: 开始录制视频 (.avi + .bin)")
    print("  [E]: 停止录制")
    print("  [S]: 截取单帧 (.png + .npy)")

    try:
        while True:
            bgr_img, c_arr, d_arr = cam.get_frames()

            key = cv2.waitKey(1) & 0xFF
            if key == 27: 
                break
            elif key == ord('r'):
                recorder.start_recording()
            elif key == ord('e'):
                recorder.stop_recording()
            elif key == ord('s'):
                recorder.save_snapshot(bgr_img, d_arr)
                cv2.putText(bgr_img, "SNAPSHOT SAVED!", (width//4, height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            recorder.record_frame(bgr_img, d_arr)

            
            # 深度图可视化 将16位转为8位伪彩色
            d_vis = np.clip(d_arr, 0, 3000) 
            d_vis = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)

            if recorder.is_recording:
                cv2.circle(bgr_img, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(bgr_img, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(bgr_img, "STANDBY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            combined_view = np.hstack((bgr_img, d_color))
            
            display_scale = 0.8
            show_img = cv2.resize(combined_view, (int(width * 2 * display_scale), int(height * display_scale)))
            
            cv2.imshow("Apple Data Recorder (Left: RGB, Right: Depth)", show_img)

    finally:
        recorder.stop_recording()
        cam.release()
        cv2.destroyAllWindows()



def test_playback():
    
    video_path = "videos/video_20260705_221009_rgb.avi"
    bin_path = "videos/video_20260705_221009_depth.bin"

    if not os.path.exists(video_path) or not os.path.exists(bin_path):
        print("找不到指定的文件，请检查路径是否正确。")
        return

    try:
        cam = VirtualCamera(video_path, bin_path, width=640, height=480)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    
    paused = False

    while True:
        if not paused:
            bgr_img, c_arr, d_arr = cam.get_frames()

            d_vis = np.clip(d_arr, 500, 3500)
            d_vis = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)

            combined = np.hstack((bgr_img, d_color))
            
            cv2.putText(combined, "Playback Mode", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    

            cv2.imshow("VirtualCamera Test (RGB | Normalized Depth)", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord(' '): 
            paused = not paused
            if paused: print("已暂停")
            else: print("继续播放")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    