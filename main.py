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

def main():
    cam_hardware = AstraCamera()
    # config.FX, config.FY, config.CX, config.CY = cam_hardware.get_intrinsics() #更新内参
    print("相机内参：", config.FX, config.FY, config.CX, config.CY)

    fps_camera = FPSCamera()
    tracker = AppleTracker()
    vision = VisionProcessor()   
    renderer = SceneRenderer() 

    print("TAB 键解锁/锁定鼠标，ESC 退出。")

    try:
        while fps_camera.running:
            bgr_img, c_arr, d_arr = cam_hardware.get_frames()

            acc_mask, target_pts_3d, params = vision.process(bgr_img, d_arr)

            raw_apples = segment_and_filter_apples(target_pts_3d, params.norm_angle, params.min_rad, params.max_rad)

            confirmed_apples = tracker.update(raw_apples, params.confirm_f, params.lost_f)
            
            if len(confirmed_apples) != fps_camera.last_apple_count:
                print(f"当前追踪 {len(confirmed_apples)} 个苹果")
                fps_camera.last_apple_count = len(confirmed_apples)

            fps_camera.update()

            renderer.update_3d_environment(c_arr, d_arr)
            renderer.update_apples(confirmed_apples)
            renderer.update_camera_view(fps_camera.get_extrinsic())
            renderer.show_2d_windows(bgr_img, d_arr, acc_mask,confirmed_apples)

            if cv2.waitKey(1) == 27: 
                break
    finally:
        print("exit")
        fps_camera.release()
        cam_hardware.release()
        renderer.release()

if __name__ == "__main__":

    main()
    