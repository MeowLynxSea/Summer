import numpy as np
from config import TRACKER_DIST_THRESH, TRACKER_EMA_ALPHA

class TrackedApple:
    def __init__(self, center, radius,cluster_pts):
        self.center = center
        self.radius = radius
        self.cluster_pts = cluster_pts 
        self.hits = 1             
        self.misses = 0           
        self.is_confirmed = False 

    def update(self, center, radius,cluster_pts):
        alpha = TRACKER_EMA_ALPHA 
        self.center = alpha * center + (1 - alpha) * self.center
        self.radius = alpha * radius + (1 - alpha) * self.radius
        self.cluster_pts = cluster_pts
        self.hits += 1
        self.misses = 0

    def predict_miss(self):
        self.misses += 1

class AppleTracker:
    def __init__(self, dist_thresh=TRACKER_DIST_THRESH):
        self.tracks = []
        self.dist_thresh = dist_thresh 

    def update(self, detections, confirm_frames, max_lost_frames):
        unmatched_detections = list(detections)
        
        for track in self.tracks:
            if not unmatched_detections:
                track.predict_miss()
                continue
                
            dists = [np.linalg.norm(track.center - d[0]) for d in unmatched_detections]
            min_idx = np.argmin(dists)
            
            if dists[min_idx] < self.dist_thresh:
                track.update(unmatched_detections[min_idx][0], 
                             unmatched_detections[min_idx][1],
                             unmatched_detections[min_idx][2])
                unmatched_detections.pop(min_idx)
            else:
                track.predict_miss()
                
        for d in unmatched_detections:
            self.tracks.append(TrackedApple(d[0], d[1],d[2]))
            
        for track in self.tracks:
            if track.hits >= confirm_frames:
                track.is_confirmed = True
                
        self.tracks = [t for t in self.tracks if t.misses <= max_lost_frames]
        return [(t.center, t.radius,t.cluster_pts) for t in self.tracks if t.is_confirmed]