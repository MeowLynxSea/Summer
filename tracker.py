import numpy as np
from config import TRACKER_DIST_THRESH, TRACKER_EMA_ALPHA,PREDICT_DAMPING

class TrackedApple:
    def __init__(self, center, radius,cluster_pts):
        self.center = center
        self.radius = radius
        self.cluster_pts = cluster_pts 
        self.hits = 1             
        self.misses = 0           
        self.is_confirmed = False 

        self.velocity = np.zeros(3) 
        self.last_center = center.copy()
    
    def predict(self):
        return self.center + self.velocity
    
    def update(self, center, radius,cluster_pts):
        alpha = TRACKER_EMA_ALPHA 

        v_alpha = 0.5 
        current_v = center - self.last_center
        self.velocity = v_alpha * current_v + (1 - v_alpha) * self.velocity
        self.last_center = center.copy()

        self.center = alpha * center + (1 - alpha) * self.center
        self.radius = alpha * radius + (1 - alpha) * self.radius
        self.cluster_pts = cluster_pts
        self.hits += 1
        self.misses = 0

    def predict_miss(self):
        self.center = self.center + self.velocity
        self.velocity *= PREDICT_DAMPING
        self.misses += 1

class AppleTracker:
    def __init__(self, dist_thresh=TRACKER_DIST_THRESH):
        self.tracks = []
        self.dist_thresh = dist_thresh 
    def update(self, detections, confirm_frames, max_lost_frames):
        num_detections = len(detections)
        num_tracks = len(self.tracks)
        
        all_possible_matches = []
        for t_idx, track in enumerate(self.tracks):
            predicted_pos = track.predict()
            
            for d_idx, det in enumerate(detections):
                det_center = det[0]
                det_radius = det[1]
                
                dist = np.linalg.norm(predicted_pos - det_center)
                
                # 重叠度判定
                spatial_overlap_limit = (track.radius + det_radius) * 0.9
                
                effective_thresh = max(self.dist_thresh, spatial_overlap_limit)
                
                if dist < effective_thresh:
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
                self.tracks[t_idx].predict_miss()

        # 没匹配上的新检测创建新追踪
        for d_idx in range(num_detections):
            if d_idx not in matched_detections:
                self.tracks.append(TrackedApple(detections[d_idx][0], 
                                               detections[d_idx][1], 
                                               detections[d_idx][2]))

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