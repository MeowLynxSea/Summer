import numpy as np
import open3d as o3d
from config import *
from scipy.spatial import cKDTree 

def fit_sphere_least_squares(pts):
    if len(pts) < 15: return None, None
    A = np.zeros((len(pts), 4))
    A[:, 0:3] = pts
    A[:, 3] = 1
    B = np.sum(pts**2, axis=1)
    try:
        res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        center = res[0:3] / 2
        r2 = res[3] + np.sum(center**2)
        if r2 < 0: return None, None
        return center, np.sqrt(r2)
    except:
        return None, None

def segment_and_filter_apples(pts_3d, norm_angle_thresh, min_radius, max_radius):
    red_pcd = o3d.geometry.PointCloud()
    red_pcd.points = o3d.utility.Vector3dVector(pts_3d)

    # 下采样减少计算压力
    red_pcd = red_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if len(red_pcd.points) < 20: return []
    
    # KDTree 加速搜索
    red_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_SEARCH_RADIUS, max_nn=NORMAL_MAX_NN))
    red_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    
    pts = np.asarray(red_pcd.points)
    normals = np.asarray(red_pcd.normals)
    
    cos_thresh = np.cos(np.radians(max(1, norm_angle_thresh))) 

    # Scipy cKDTree 并行过滤
    # tree = cKDTree(pts)
    # _, indices = tree.query(pts, k=8, workers=-1)
    # neighbor_normals = normals[indices]
    # dots = np.einsum('ij,ikj->ik', normals, neighbor_normals)
    # passing_neighbors = np.sum(dots > cos_thresh, axis=1)
    # mask = passing_neighbors >= PASSING_NEIGHBORS

    # valid_indices = np.where(mask)[0]
    # if len(valid_indices) < SPHERE_FIT_MIN_PTS: return []
    # filtered_pcd = red_pcd.select_by_index(valid_indices)


    invalid_set = set()
    kdtree = o3d.geometry.KDTreeFlann(red_pcd)  # 建立法向树
    
    for i in range(len(pts)):
        if i in invalid_set: continue
        [k, idx, _] = kdtree.search_radius_vector_3d(pts[i], KDTREE_SEARCH_RADIUS)
        if k > 1:
            dots = np.dot(normals[idx[1:]], normals[i])
            bad_local_idx = np.where(dots < cos_thresh)[0]
            if len(bad_local_idx) > 0:
                invalid_set.add(i)
                for bi in bad_local_idx:
                    invalid_set.add(idx[bi + 1])
                    
    valid_indices = [i for i in range(len(pts)) if i not in invalid_set]
    if len(valid_indices) < 20: return []
    filtered_pcd = red_pcd.select_by_index(valid_indices)

    # DBSCAN 聚类
    labels = np.array(filtered_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
    if len(labels) == 0 or labels.max() < 0: return []
    
    results = []
    for c in range(labels.max() + 1):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) < SPHERE_FIT_MIN_PTS: 
            continue
        cluster_pts = np.asarray(filtered_pcd.points)[cluster_idx]
        center, radius = fit_sphere_least_squares(cluster_pts)
        
        if center is not None and min_radius < radius < max_radius:
            results.append((center, radius, cluster_pts))
            
    return results