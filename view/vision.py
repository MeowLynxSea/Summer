import cv2
import numpy as np
from collections import namedtuple
import config
from scipy.spatial import cKDTree

VisionParams = namedtuple("VisionParams", ["norm_angle","min_rad", "max_rad", "confirm_f", "lost_f"])

class VisionProcessor:
    def __init__(self):
        self.window_name = "Red Extraction"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self._create_trackbars()
        self.mask_history = []

    def _create_trackbars(self):
        
        cv2.createTrackbar("Mode 0:HSV 1:LAB", self.window_name, config.UI_DEF_COLOR_MODE, 1, lambda x: None)   
        cv2.createTrackbar("Geo Mask Enable", self.window_name, config.UI_DEF_GEO_ENABLE, 1, lambda x: None)
        
        cv2.createTrackbar("Geo Sphere Min", self.window_name, config.UI_DEF_SPHERICITY_MIN, 15, lambda x: None)
        cv2.createTrackbar("Geo Planar Max", self.window_name, config.UI_DEF_PLANARITY_MAX, 100, lambda x: None)
        cv2.createTrackbar("Geo Saddle Ratio", self.window_name, config.UI_DEF_SADDLE_RATIO, 50, lambda x: None)

        cv2.createTrackbar("LAB A Min", self.window_name, config.UI_DEF_LAB_A_MIN, 255, lambda x: None)
        cv2.createTrackbar("LAB L Min", self.window_name, config.UI_DEF_LAB_L_MIN, 255, lambda x: None)


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

    
    def _get_ui_params(self):
        p = {}
        p['color_mode'] = cv2.getTrackbarPos("Mode 0:HSV 1:LAB", self.window_name)
        p['geo_enable'] = cv2.getTrackbarPos("Geo Mask Enable", self.window_name)
        p['geo_sphere_min'] = cv2.getTrackbarPos("Geo Sphere Min", self.window_name) / 100.0
        p['geo_planar_max'] = cv2.getTrackbarPos("Geo Planar Max", self.window_name) / 100.0
        p['geo_saddle_ratio'] = cv2.getTrackbarPos("Geo Saddle Ratio", self.window_name) / 100.0
        p['lab_a_min'] = cv2.getTrackbarPos("LAB A Min", self.window_name)
        p['lab_l_min'] = cv2.getTrackbarPos("LAB L Min", self.window_name)
        p['hue_tol'] = cv2.getTrackbarPos("Hue Tol", self.window_name)
        p['sat_min'] = cv2.getTrackbarPos("Sat Min", self.window_name)
        p['val_min'] = cv2.getTrackbarPos("Val Min", self.window_name)
        p['time_win'] = max(1, cv2.getTrackbarPos("Time Win", self.window_name))
        p['norm_angle'] = cv2.getTrackbarPos("Norm Angle", self.window_name)
        p['min_rad'] = cv2.getTrackbarPos("Min Rad", self.window_name) / 1000.0
        p['max_rad'] = cv2.getTrackbarPos("Max Rad", self.window_name) / 1000.0
        p['confirm_f'] = max(1, cv2.getTrackbarPos("Confirm Frm", self.window_name))
        p['lost_f'] = cv2.getTrackbarPos("Lost Frm", self.window_name)
        
        # 卷积核为奇数
        bk = cv2.getTrackbarPos("Blur K", self.window_name)
        p['blur_k'] = bk + 1 if bk % 2 == 0 else bk
        p['blur_k'] = max(1, p['blur_k'])
        
        p['m_open'] = cv2.getTrackbarPos("Morph Open", self.window_name)
        p['m_close'] = cv2.getTrackbarPos("Morph Close", self.window_name)
        p['iter_open'] = cv2.getTrackbarPos("Iter Morph Open", self.window_name)
        p['iter_close'] = cv2.getTrackbarPos("Iter Morph Close", self.window_name)
        return p
    
    
    def _color_segmentation(self, bgr_img, p):
        blurred = cv2.GaussianBlur(bgr_img, (p['blur_k'], p['blur_k']), 0)
        
        if p['color_mode'] == 0:
            # HSV
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0, p['sat_min'], p['val_min']]), np.array([p['hue_tol'], 255, 255]))
            m2 = cv2.inRange(hsv, np.array([180 - p['hue_tol'], p['sat_min'], p['val_min']]), np.array([180, 255, 255]))
            return cv2.bitwise_or(m1, m2)
        else:
            # LAB
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
            L, a, b = cv2.split(lab)
            _, a_m = cv2.threshold(a, p['lab_a_min'], 255, cv2.THRESH_BINARY)
            _, l_m = cv2.threshold(L, p['lab_l_min'], 255, cv2.THRESH_BINARY)
            return cv2.bitwise_and(a_m, l_m)
        
    def _geometric_segmentation(self, d_arr, p):
        if not p['geo_enable']:
            return np.zeros_like(d_arr, dtype=np.uint8)

        skip = 3 # 跳采样，加速
        h, w = d_arr.shape
        y, x = np.mgrid[0:h:skip, 0:w:skip]
        z = d_arr[0:h:skip, 0:w:skip] / 1000.0
        
        mask_z = (z > config.DEPTH_Z_MIN) & (z < config.DEPTH_Z_MAX)
        if not np.any(mask_z): return np.zeros_like(d_arr, dtype=np.uint8)
        
        z_v = z[mask_z]
        x_v = (x[mask_z] - config.CX) * z_v / config.FX
        y_v = -(y[mask_z] - config.CY) * z_v / config.FY
        pts = np.stack((x_v, y_v, z_v), axis=-1)
        
        tree = cKDTree(pts)
        neighbors_list = tree.query_ball_point(pts, r=config.NORMAL_SEARCH_RADIUS , workers=-1)
        
        candidate_coords = []
        
        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) < 10: continue

            local_pts = pts[neighbors]
            diff = local_pts - np.mean(local_pts, axis=0)
            cov = np.dot(diff.T, diff) / len(neighbors)
            
            # 特征值分解
            evals, evecs = np.linalg.eigh(cov) # L3 <= L2 <= L1
            L3, L2, L1 = evals
            normal = evecs[:, 0] 

            # pca曲率 
            curvature = L3 / (L1 + L2 + L3)
            # 线度 
            linearity = L2 / L1

            # 凸性
            center_pt = pts[i]
            vecs = local_pts - center_pt
            dists = np.dot(vecs, normal)
            pos_count = np.sum(dists > 0.001)
            neg_count = np.sum(dists < -0.001)
            side_ratio = min(pos_count, neg_count) / len(neighbors)

            if side_ratio > p['geo_saddle_ratio']: # 如果超过ratio的点在另一侧，为非纯凸面
                continue

            if p['geo_sphere_min'] < curvature < 0.15 and linearity < p['geo_planar_max']:
                candidate_coords.append((y[mask_z][i], x[mask_z][i]))

        geo_mask_2d = np.zeros_like(d_arr, dtype=np.uint8)
        if len(candidate_coords) > 0:
            coords = np.array(candidate_coords)
            geo_mask_2d[coords[:, 0], coords[:, 1]] = 255
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (skip*2+1, skip*2+1))
            geo_mask_2d = cv2.dilate(geo_mask_2d, kernel)
                
        return geo_mask_2d
    
    
    def _refine_mask(self, mask, p):
        if p['m_open'] > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (p['m_open'], p['m_open']), iterations=p['iter_open'])
        if p['m_close'] > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (p['m_close'], p['m_close']), iterations=p['iter_close'])
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask)
            
        areas = stats[:, cv2.CC_STAT_AREA]
        keep_labels = np.where(areas > config.MIN_CONTOUR_AREA)[0]
        keep_labels = keep_labels[keep_labels != 0]
        
        return np.isin(labels, keep_labels).astype(np.uint8) * 255
    
    def _project_to_3d(self, mask, d_arr):
        y_idx, x_idx = np.where(mask > 0)
        if len(y_idx) == 0:
            return []
            
        z_v = d_arr[y_idx, x_idx] / 1000.0 
        valid_z = (z_v > config.DEPTH_Z_MIN) & (z_v < config.DEPTH_Z_MAX)
        
        if not np.any(valid_z):
            return []
            
        z_v = z_v[valid_z]
        x_v = (x_idx[valid_z] - config.CX) * z_v / config.FX
        y_v = -(y_idx[valid_z] - config.CY) * z_v / config.FY
        
        return np.stack((x_v, y_v, z_v), axis=-1)
    
    def process(self, bgr_img, d_arr):
        p = self._get_ui_params()

        red_mask = self._color_segmentation(bgr_img, p)
        geo_mask = self._geometric_segmentation(d_arr, p)
        combined_mask = cv2.bitwise_or(red_mask, geo_mask)

        refined_mask = self._refine_mask(combined_mask, p)

        self.mask_history.append(refined_mask)
        if len(self.mask_history) > p['time_win']:
            self.mask_history.pop(0)
        acc_mask = np.bitwise_or.reduce(self.mask_history)

        target_pts_3d = self._project_to_3d(acc_mask, d_arr)

        params = VisionParams(p['norm_angle'], p['min_rad'], p['max_rad'], p['confirm_f'], p['lost_f'])
        
        return acc_mask, target_pts_3d, params