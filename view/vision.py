import cv2
import numpy as np
from collections import namedtuple
import config

VisionParams = namedtuple("VisionParams", ["norm_angle","min_rad", "max_rad", "confirm_f", "lost_f"])

class VisionProcessor:
    def __init__(self):
        self.window_name = "Red Extraction"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self._create_trackbars()
        self.mask_history = []

    def _create_trackbars(self):
        cv2.createTrackbar("Hue Tol", self.window_name, config.UI_DEF_HUE_TOL, 30, lambda x: None)
        cv2.createTrackbar("Sat Min", self.window_name, config.UI_DEF_SAT_MIN, 255, lambda x: None)
        cv2.createTrackbar("Val Min", self.window_name, config.UI_DEF_VAL_MIN, 255, lambda x: None)
        cv2.createTrackbar("Time Win", self.window_name, config.UI_DEF_TIME_WIN, 15, lambda x: None)   
        cv2.createTrackbar("Norm Angle", self.window_name, config.UI_DEF_NORM_ANGLE, 30, lambda x: None) 
        cv2.createTrackbar("Min Rad", self.window_name, config.UI_DEF_MIN_RAD, config.UI_DEF_MAX_RAD, lambda x: None)
        cv2.createTrackbar("Max Rad", self.window_name, 40, config.UI_DEF_MAX_RAD, lambda x: None)
        cv2.createTrackbar("Confirm Frm", self.window_name, config.UI_DEF_CONFIRM_FRM, 20, lambda x: None)
        cv2.createTrackbar("Lost Frm", self.window_name, config.UI_DEF_LOST_FRM, 30, lambda x: None)

        cv2.createTrackbar("Blur K", self.window_name, config.UI_DEF_BLUR_R, 20, lambda x: None)
        cv2.createTrackbar("Morph Open", self.window_name, config.MORPH_OPEN_KERNEL_R, 20, lambda x: None)
        cv2.createTrackbar("Morph Close", self.window_name, config.MORPH_CLOSE_KERNEL_R, 20, lambda x: None)
        cv2.createTrackbar("Iter Morph Open", self.window_name, config.ITER_MORPH_OPEN, 10, lambda x: None)
        cv2.createTrackbar("Iter Morph Close", self.window_name, config.ITER_MORPH_CLOSE, 10, lambda x: None)

    def process(self, bgr_img, d_arr):
        hue_tol = cv2.getTrackbarPos("Hue Tol", self.window_name)
        sat_min = cv2.getTrackbarPos("Sat Min", self.window_name)
        val_min = cv2.getTrackbarPos("Val Min", self.window_name)
        time_win = max(1, cv2.getTrackbarPos("Time Win", self.window_name))
        
        norm_angle = cv2.getTrackbarPos("Norm Angle", self.window_name)
        
        min_rad = cv2.getTrackbarPos("Min Rad", self.window_name) / 1000.0
        max_rad = cv2.getTrackbarPos("Max Rad", self.window_name) / 1000.0
        confirm_f = max(1, cv2.getTrackbarPos("Confirm Frm", self.window_name))
        lost_f = cv2.getTrackbarPos("Lost Frm", self.window_name)

        
        blur_k_val = cv2.getTrackbarPos("Blur K", self.window_name)
        morph_open_val = cv2.getTrackbarPos("Morph Open", self.window_name)
        morph_close_val = cv2.getTrackbarPos("Morph Close", self.window_name)
        iter_morph_open = cv2.getTrackbarPos("Iter Morph Open", self.window_name)
        iter_morph_close = cv2.getTrackbarPos("Iter Morph Close", self.window_name)

        if blur_k_val == 0:
            blur_k = 1
        elif blur_k_val % 2 == 0:
            blur_k = blur_k_val + 1
        else:
            blur_k = blur_k_val

        # 颜色提取
        hsv_img = cv2.cvtColor(cv2.GaussianBlur(bgr_img, (blur_k, blur_k), 0), cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_img, np.array([0, sat_min, val_min]), np.array([hue_tol, 255, 255]))
        mask2 = cv2.inRange(hsv_img, np.array([180 - hue_tol, sat_min, val_min]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, (morph_open_val,morph_open_val), iterations=iter_morph_open)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, (morph_close_val,morph_close_val), iterations=iter_morph_close)

        # for优化
        # contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # final_mask = np.zeros_like(red_mask)
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > config.MIN_CONTOUR_AREA: 
        #         cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        keep_labels = np.where(areas > config.MIN_CONTOUR_AREA)[0]
        keep_labels = keep_labels[keep_labels != 0]
        final_mask = np.isin(labels, keep_labels).astype(np.uint8) * 255


        # 时累积
        self.mask_history.append(final_mask)
        if len(self.mask_history) > time_win: 
            self.mask_history.pop(0)
        acc_mask = np.bitwise_or.reduce(self.mask_history) if len(self.mask_history) > 1 else self.mask_history[0]

        # 提取有效点云
        y_idx, x_idx = np.where(acc_mask > 0)
        target_pts_3d = []
        if len(y_idx) > 0:
            z_v = d_arr[y_idx, x_idx] / 1000.0
            valid_obj = (z_v > config.DEPTH_Z_MIN) & (z_v < config.DEPTH_Z_MAX) 
            if np.any(valid_obj):
                z_v = z_v[valid_obj]
                x_v = (x_idx[valid_obj] -config.CX) * z_v / config.FX
                y_v = -(y_idx[valid_obj] - config.CY) * z_v / config.FY
                target_pts_3d = np.stack((x_v, y_v, z_v), axis=-1)

        params = VisionParams(norm_angle,min_rad, max_rad, confirm_f, lost_f)
        return acc_mask, target_pts_3d, params