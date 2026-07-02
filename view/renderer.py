import cv2
import numpy as np
import open3d as o3d
import config


class SceneRenderer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Geometric Apple Tracker", width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT)
        self.ctr = self.vis.get_view_control()

        self.ctr.set_constant_z_near(0.001)
        self.ctr.set_constant_z_far(100.0)
 

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.zeros((config.IMG_WIDTH * config.IMG_HEIGHT, 3)))
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((config.IMG_WIDTH * config.IMG_HEIGHT, 3)))
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
        
        unit_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=config.APPLE_SHELL_RESOLUTION)
        unit_ls = o3d.geometry.LineSet.create_from_triangle_mesh(unit_mesh)
        
        self.template_shell_vertices = np.asarray(unit_ls.points).copy()
        self.template_shell_lines = unit_ls.lines

        self.cluster_colors = [
            (255, 0, 0), 
            (0, 255, 0),  
            (0, 255, 255), 
            (255, 0, 255), 
            (255, 255, 0), 
            (0, 165, 255),
        ]


        self.apple_centers, self.apple_shells = [], []
        for _ in range(config.MAX_APPLES):
            c = o3d.geometry.TriangleMesh.create_sphere(radius=config.APPLE_CENTER_RADIUS)
            c.paint_uniform_color([0, 1, 0]) 
            c.translate([0, 0, -100])
            self.vis.add_geometry(c)
            self.apple_centers.append(c)
            
            ls = o3d.geometry.LineSet()
            ls.lines = self.template_shell_lines
            ls.paint_uniform_color([1, 0.4, 0.4])
            ls.translate([0, 0, -100])
            self.vis.add_geometry(ls)
            self.apple_shells.append(ls)

    def update_3d_environment(self, c_arr, d_arr):
        z = d_arr.flatten() / 1000.0
        x = (config.U - config.CX) * z / config.FX
        y = -(config.V -config.CY) * z /config. FY
            
        valid_z_mask = z > 0 
        self.pcd.points = o3d.utility.Vector3dVector(np.stack((x[valid_z_mask], y[valid_z_mask], z[valid_z_mask]), axis=-1))
        self.pcd.colors = o3d.utility.Vector3dVector((c_arr.reshape(-1, 3) / 255.0)[valid_z_mask])
        self.vis.update_geometry(self.pcd)

    def update_apples(self, confirmed_apples):
        for i in range(config.MAX_APPLES):
            shell = self.apple_shells[i]
            center_geo = self.apple_centers[i]
            
            if i < len(confirmed_apples):
                center, radius,_ = confirmed_apples[i]
                center_geo.translate(center, relative=False)
                
                new_points = self.template_shell_vertices * radius + center
                shell.points = o3d.utility.Vector3dVector(new_points)
            else:
                center_geo.translate([0, 0, -100], relative=False)
                shell.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            
            self.vis.update_geometry(center_geo)
            self.vis.update_geometry(shell)

    def update_camera_view(self, extrinsic_matrix):
        vp = self.ctr.convert_to_pinhole_camera_parameters()
        vp.extrinsic = extrinsic_matrix
        self.ctr.convert_from_pinhole_camera_parameters(vp, True)
        self.vis.poll_events()
        self.vis.update_renderer()

    def show_2d_windows(self, bgr_img, d_arr, acc_mask,confirmed_apples):
        red_extracted_img = cv2.bitwise_and(bgr_img, bgr_img, mask=acc_mask)
        
        render_max_mm = config.DEPTH_Z_MAX * 1000.0 
        
        depth_colormap = cv2.applyColorMap(
            (np.clip(d_arr, 0, render_max_mm) / render_max_mm * 255.0).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        depth_colormap[d_arr == 0] = [0, 0, 0] 
        
        seg_debug_img = bgr_img.copy()

        for i, (center, radius, cluster_pts) in enumerate(confirmed_apples):
     
            color = self.cluster_colors[i % len(self.cluster_colors)]
            
            pts_x = cluster_pts[:, 0]
            pts_y = cluster_pts[:, 1]
            pts_z = cluster_pts[:, 2]
            
            u_pts = (pts_x * config.FX / pts_z + config.CX).astype(np.int32)
            v_pts = (config.CY - pts_y * config.FY / pts_z).astype(np.int32)
            
            for u, v in zip(u_pts, v_pts):
                if 0 <= u < config.IMG_WIDTH and 0 <= v < config.IMG_HEIGHT:
                    cv2.circle(seg_debug_img, (u, v), 2, color, -1)

            cz = center[2]
            cu = int(center[0] * config.FX / cz + config.CX)
            cv = int(config.CY  - center[1] * config.FY / cz)
            cv2.putText(seg_debug_img, f"Apple_{i}", (cu, cv), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cv2.imshow("Segmentation Result", cv2.flip(seg_debug_img, 1))
        cv2.imshow("Astra RGB", cv2.flip(bgr_img, 1))
        cv2.imshow("Astra Depth", cv2.flip(depth_colormap, 1))
        cv2.imshow("Red Extraction", cv2.flip(red_extracted_img, 1))
    
    def release(self):
        self.vis.destroy_window()
        cv2.destroyAllWindows()