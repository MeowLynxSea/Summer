import numpy as np
from config import *


class TrackedApple:
    def __init__(self, center, radius, cluster_pts, dt):
        self.cluster_pts = cluster_pts
        self.hits = 1
        self.misses = 0
        self.is_confirmed = False

        # 卡尔曼滤波初始化
        # X: [x, y, z, vx, vy, vz, r]
        self.X = np.array([center[0], center[1], center[2],
                          0, 0, 0, radius], dtype=np.float32)

        # F
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        self.H[3, 6] = 1

        self.P = np.eye(7, dtype=np.float32) * 1.0

        self.Q = np.eye(7, dtype=np.float32) * KF_PROC_NOISE

        self.R = np.diag([KF_POS_NOISE, KF_POS_NOISE,
                         KF_POS_NOISE, KF_RAD_NOISE]).astype(np.float32)

    @property
    def center(self):
        return self.X[:3]

    @property
    def radius(self):
        return self.X[6]
    @property
    def speed(self):
        return np.linalg.norm(self.X[3:6])
    
    def predict(self, dt):

        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X[:3]

    def update(self, center, radius, cluster_pts):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        Z = np.array([center[0], center[1], center[2], radius],
                     dtype=np.float32)
        y = Z - np.dot(self.H, self.X)
        self.X = self.X + np.dot(K, y)

        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

        self.cluster_pts = cluster_pts
        self.hits += 1
        self.misses = 0

    def predict_miss(self, dt):
        self.predict(dt)
        self.X[3:6] *= PREDICT_DAMPING
        self.misses += 1


class AppleTracker:
    def __init__(self, dist_thresh=TRACKER_DIST_THRESH):
        self.tracks = []
        self.dist_thresh = dist_thresh

    def update(self, detections, confirm_frames, max_lost_frames, dt):
        num_detections = len(detections)
        num_tracks = len(self.tracks)

        all_possible_matches = []
        for t_idx, track in enumerate(self.tracks):
            predicted_pos = track.predict(dt)
            current_speed = track.speed

            for d_idx, det in enumerate(detections):
                det_center = det[0]
                det_radius = det[1]

                dist = np.linalg.norm(predicted_pos - det_center)

                # 重叠度判定
                spatial_overlap_limit = (track.radius + det_radius) * 0.9
                speed_allowance = current_speed * dt * 2.0

                gating_radius = max(self.dist_thresh, spatial_overlap_limit + speed_allowance)
                
                if track.is_confirmed:
                    gating_radius *= 1.2

                if dist < gating_radius:
                    all_possible_matches.append({
                        'track_idx': t_idx,
                        'det_idx': d_idx,
                        'dist': dist
                    })

        # 全局贪婪匹配
        all_possible_matches.sort(key=lambda x: x['dist'])

        matched_tracks = set()
        matched_detections = set()

        for match in all_possible_matches:
            t_idx = match['track_idx']
            d_idx = match['det_idx']

            if t_idx not in matched_tracks and d_idx not in matched_detections:
                self.tracks[t_idx].update(detections[d_idx][0],
                                          detections[d_idx][1],
                                          detections[d_idx][2])
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)

        # 没匹配上的旧目标标记丢失
        for t_idx in range(num_tracks):
            if t_idx not in matched_tracks:
                self.tracks[t_idx].predict_miss(dt)

        # 没匹配上的新检测创建新追踪
        for d_idx in range(num_detections):
            if d_idx not in matched_detections:
                self.tracks.append(TrackedApple(detections[d_idx][0],
                                                detections[d_idx][1],
                                                detections[d_idx][2],
                                                dt))

        for track in self.tracks:
            if track.hits >= confirm_frames:
                track.is_confirmed = True

        self.tracks = [t for t in self.tracks if t.misses <= max_lost_frames]

        return [(t.center, t.radius, t.cluster_pts) for t in self.tracks if t.is_confirmed]

    # def update(self, detections, confirm_frames, max_lost_frames):
    #     unmatched_detections = list(detections)

    #     for track in self.tracks:
    #         if not unmatched_detections:
    #             track.predict_miss()
    #             continue

    #         predicted_pos = track.predict()
    #         dists = [np.linalg.norm(predicted_pos - d[0]) for d in unmatched_detections]
    #         min_idx = np.argmin(dists)

    #         if dists[min_idx] < self.dist_thresh:
    #             track.update(unmatched_detections[min_idx][0],
    #                          unmatched_detections[min_idx][1],
    #                          unmatched_detections[min_idx][2])
    #             unmatched_detections.pop(min_idx)
    #         else:
    #             track.predict_miss()

    #     for d in unmatched_detections:
    #         self.tracks.append(TrackedApple(d[0], d[1],d[2]))

    #     for track in self.tracks:
    #         if track.hits >= confirm_frames:
    #             track.is_confirmed = True

    #     self.tracks = [t for t in self.tracks if t.misses <= max_lost_frames]
    #     return [(t.center, t.radius,t.cluster_pts) for t in self.tracks if t.is_confirmed]
