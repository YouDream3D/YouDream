import os
import os.path as osp
import cv2
import math
import imageio
import torch
import torch.utils.dlpack
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pickle
import os.path as osp
import open3d as o3d
from typing import Iterable

import json

from configs.train_config import PromptConfig
from core.utils.point3d import *
from core.utils.pose import index2pose, SE3_Mat2RT, NeRF_data_to_standard

def write_obj(vertices: np.ndarray, filename: str):
    with open(filename, 'w') as file:
        file.write("# OBJ file generated\n")
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

def read_obj_file_and_extract_vertices(file_path):
    vertices = []  # List to store vertices
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'v':  # Identifies a vertex line
                vertex = [float(coord) for coord in parts[1:4]]  # Convert the coordinates to float and store
                vertices.append(vertex)
    return vertices

class MyPoseSkeleton(object):
    def __init__(self, cfg) -> None:
        
        self.cfg = cfg

    def __call__(self):
        """
        Input:
        Return:
            vertices: np.array, shape = (N, V, 3)
            joints: np.array, shape = (N, J, 3)
            keypoints: np.array, shape = (N, 18, 3)
        """
        vertices = []
        faces = []
        with open(self.cfg.init_mesh, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts and parts[0] == 'v':  # Only process vertices lines
                    x, y, z = map(float, parts[1:4])
                    vertices.append([x, y, z])
                if parts and parts[0] == 'f':  # Only process vertices lines
                    x = int(parts[1].split('//')[0]) - 1
                    y = int(parts[2].split('//')[0]) - 1
                    z = int(parts[3].split('//')[0]) - 1 
                    # x, y, z = map(float, parts[1:4])
                    faces.append([x, y, z])
        
        scale_cfg = self.cfg.scale_init
        vertices = np.array(vertices)[None, :, :]
        faces = np.array(faces)

        ap10k_keypoints = []
        colors_joints = []
        with open(self.cfg.init_pose, 'r') as json_file:
            data = json.load(json_file)
            keypoints = data['keypoints']
            for keypoint in keypoints:
                position = keypoint['position']
                x, y, z = float(position['x']), float(position['y']), float(position['z'])
                ap10k_keypoints.append([x, y, z])
                colors_joints.append(keypoint['color'])
            
            self.colors_joints = colors_joints
            self.limbSeq = data['limbSeq']
            self.colors_limbs = data['colors_limbs']
            self.face_left_indices = data["face_left_indices"]
            self.face_right_indices = data["face_right_indices"]
            self.face_nose_indices = data["face_nose_indices"]

        ap10k_keypoints = np.array(ap10k_keypoints)[None, :, :]

        joints = None

        # Rotation matrix for xz-plane (around z) by 0 degrees
        R_xz = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

        # Rotation matrix for xy-plane (around z) by 0 degrees
        R_xy = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

        ap10k_keypoints_mean = ap10k_keypoints.mean(axis=1)

        shift = np.mean(ap10k_keypoints,axis=1)
        ap10k_keypoints = ap10k_keypoints-np.repeat(shift[:, None, :], ap10k_keypoints.shape[1], axis=1)

        scale = np.max(np.linalg.norm(ap10k_keypoints, ord=2, axis=1))
        ap10k_keypoints = ap10k_keypoints/scale 
        ap10k_keypoints[0,:,0] = -1*ap10k_keypoints[0,:,0]


        vertices = vertices - np.repeat(shift[:, None, :], vertices.shape[1], axis=1)
        vertices = vertices/scale 
        vertices[0,:,0] = -1*vertices[0,:,0]

        vertices = vertices * scale_cfg
        ap10k_keypoints = ap10k_keypoints * scale_cfg
        
        joints = vertices.copy()

        return vertices, joints, ap10k_keypoints, faces 

def draw_keypoints_on_blank(keypt_dict, colors_joints, limbSeq, colors_limbs, image_width=512, image_height=512, point_radius=4, line_width=5):
    # Create a new image with white background
    image = Image.new('RGB', (image_width, image_height), 'black')
    draw = ImageDraw.Draw(image)

    # Draw lines for each bone in the sequence
    for (start_kpt, end_kpt), color in zip(limbSeq, colors_limbs):
        start_point = keypt_dict[start_kpt]
        end_point = keypt_dict[end_kpt]
        # Check if both points are valid
        if is_nan(start_point[0]) or is_nan(start_point[1]) or is_nan(end_point[0]) or is_nan(end_point[1]):
            continue
        x1, y1 = round(start_point[0]), round(start_point[1])
        x2, y2 = round(end_point[0]), round(end_point[1])
        # Draw the line
        draw.line([x1, y1, x2, y2], fill=color, width=line_width)

    # Iterate through the joints
    for kpt, color in zip(keypt_dict, colors_joints):
        # Check if the point is valid (not [-1, -1])
        x, y = kpt
        if is_nan(x) or is_nan(y):
            continue
        # Round the coordinates to the nearest integer
        x = round(x)
        y = round(y)
        # Draw the point as a circle
        left_up_point = (x - point_radius, y - point_radius)
        right_down_point = (x + point_radius, y + point_radius)
        draw.ellipse([left_up_point, right_down_point], fill=color)

    # Return the image with keypoints
    return image

class _HumanScene(object):
    
    def build_scene(self):
        meshs = []
        ray_casting_scene = o3d.t.geometry.RaycastingScene()
        for each_vertices in self.vertices:
            mesh = o3d.geometry.TriangleMesh(
                vertices = o3d.utility.Vector3dVector(each_vertices),
                triangles = o3d.utility.Vector3iVector(self.triangles),
            )
            mesh.compute_vertex_normals()
            meshs.append(mesh)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            ray_casting_scene.add_triangles(mesh_t)
        return meshs, ray_casting_scene

    def export_depth_map(self, intrinsics, extrinsic, width=512, height=512, inverse=True, normalize=True):
        """
        Input:
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        """
        # Rays are 6D vectors with origin and ray direction.
        # Here we use a helper function to create rays for a pinhole camera.
        rays = self.ray_casting_scene.create_rays_pinhole(intrinsics, extrinsic, width_px=width, height_px=height)

        # Compute the ray intersections.
        ans = self.ray_casting_scene.cast_rays(rays)
        depth = ans['t_hit'].numpy()

        # Inverse and Normalize
        if inverse:
            depth = 1.0 / depth
        if normalize:
            depth -= np.min(depth)
            depth /= np.max(depth)
        image = np.asarray(depth * 255.0, np.uint8)
        image = np.stack([image, image, image], axis=2)
        return Image.fromarray(image)

    def export_pose_map(self, intrinsics, extrinsic, dirs, width=512, height=512, occlusion_culling=False):
        """
        Input:
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        Variable:
            self.keypoints: np.array, [N, K, 3]
        """
        # Init
        N, K, _ = self.keypoints.shape
        R, T = SE3_Mat2RT(extrinsic)
        # Transform
        kp_world = self.keypoints.reshape(-1, 3)
        kp_camera = transform_keypoints_to_novelview(kp_world, None, None, R, T)
        kp_image = project_camera3d_to_2d(kp_camera, intrinsics)  # [N*18, 2]
        kp_image = kp_image.reshape(N, K, 2)  # [N, 18, 2]
        # Occlusion
        if occlusion_culling:
            if dirs.item() == 1: 
                for index in self.PoseSkeleton.face_left_indices:
                    kp_image[:, index, :] = None 

            if dirs.item() == 2: 
                for index in self.PoseSkeleton.face_left_indices:
                    kp_image[:, index, :] = None 

                for index in self.PoseSkeleton.face_right_indices:
                    kp_image[:, index, :] = None 

                for index in self.PoseSkeleton.face_nose_indices:
                    kp_image[:, index, :] = None 

            if dirs.item() == 3: 
                for index in self.PoseSkeleton.face_right_indices:
                    kp_image[:, index, :] = None 
                
        # Draw
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        image = self.draw_bodypose(canvas, kp_image)
        return image

    def export_mesh_map(self, intrinsics, extrinsic, width=512, height=512, focal=512.0, device=None):
        import pytorch3d
        import pytorch3d.renderer
        from scipy.spatial.transform import Rotation

        ''' Render the mesh under camera coordinates
        vertices: (N_v, 3), vertices of mesh
        faces: (N_f, 3), faces of mesh
        translation: (3, ), translations of mesh or camera
        focal: float, focal length of camera
        height: int, height of image
        width: int, width of image
        device: "cpu"/"cuda:0", device of torch
        :return: the rgba rendered image
        '''

        if device is None:
            device = torch.device('cuda')

        vertices = torch.from_numpy(self.vertices[0]).to(device)
        faces = torch.from_numpy(self.triangles.astype(np.int64)).to(device)

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)[None]  # (B, V, 3)
        textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
        mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces], textures=textures)

        focal = intrinsics[0][0].item()

        R = torch.from_numpy(extrinsic[np.newaxis, :3, :3])
        T = torch.from_numpy(extrinsic[np.newaxis, :3, 3])
        R = R.transpose(1,2)
        R[:,:,0:2] = -R[:,:,0:2] # y = -y, z = -z

        # print(R.shape)  # [4, 4]
        # print(T.shape)  # [4, 4]

        if not hasattr(self, 'mesh_renderer'):
            # Define the settings for rasterization and shading.
            raster_settings = pytorch3d.renderer.RasterizationSettings(
                # image_size=(height, width),   # (H, W)
                image_size=height,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # Define the material
            materials = pytorch3d.renderer.Materials(
                ambient_color=((1, 1, 1),),
                diffuse_color=((1, 1, 1),),
                specular_color=((1, 1, 1),),
                shininess=64,
                device=device
            )

            # Place a directional light in front of the object.
            lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 2, 3),))
            # lights = pytorch3d.renderer.AmbientLights(ambient_color=((1.0, 1.0, 1.0),), device=device)

            # Create a phong renderer by composing a rasterizer and a shader.
            renderer = pytorch3d.renderer.MeshRenderer(
                rasterizer=pytorch3d.renderer.MeshRasterizer(
                    raster_settings=raster_settings
                ),
                shader=pytorch3d.renderer.SoftPhongShader(
                    device=device,
                    lights=lights,
                    materials=materials
                )
            )

            self.mesh_renderer = renderer

        # Initialize a camera.
        # R: Rotation matrix of shape (N, 3, 3)
        # T: Translation matrix of shape (N, 3)
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=(
                (2 * focal / min(height, width), 2 * focal / min(height, width)),
            ),
            R=R,
            T=T,
            image_size=((height, width),),
            device=device,
        )

        # Do rendering
        color_batch = self.mesh_renderer(mesh, cameras=cameras)  # [1, 512, 512, 4]

        # To Image
        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()
        input_img = np.zeros_like(color[:, :, :3])
        alpha = 1.0
        image_vis = alpha * color[:, :, :3] * valid_mask + (1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        image_vis = image_vis.astype(np.uint8)

        image = Image.fromarray(image_vis, mode='RGB')
        return image

    def export_distance(self, query_points: torch.Tensor, signed=True):
        """
        Input:
            query_points: torch.Tensor, [..., 3]
        Return:
            distances: torch.Tensor, [...]
        """
        if isinstance(query_points, torch.Tensor):
            device = query_points.device
            query_points = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query_points.detach().cpu()))
        if signed:
            distances = self.ray_casting_scene.compute_signed_distance(query_points)
        else:
            distances = self.ray_casting_scene.compute_distance(query_points)
        distances = torch.utils.dlpack.from_dlpack(distances.to_dlpack()).to(device)
        return distances

    def draw_bodypose(self, canvas, keypoints_2d):
        """
        canvas = np.zeros_like(input_image), np.array, [H x W x 3]
        keypoints_2d: np.array, [N, 17, 2], N is the number of people
        """
        limbSeq = self.PoseSkeleton.limbSeq
        colors_limbs = self.PoseSkeleton.colors_limbs
        colors_joints = self.PoseSkeleton.colors_joints
        
        # assert keypoints_2d.shape[1] == 18 and keypoints_2d.ndim in (2, 3)
        if keypoints_2d.ndim == 2:
            keypoints_2d = keypoints_2d[np.newaxis, ...]
            
        N = keypoints_2d.shape[0]
        for p in range(N):
            image = draw_keypoints_on_blank(keypt_dict=keypoints_2d[0], colors_joints=colors_joints, limbSeq=limbSeq, colors_limbs=colors_limbs, image_width=canvas.shape[0], image_height=canvas.shape[0], point_radius=4, line_width=5)
            
        # return Image.fromarray(image)
        return image

    def export_geometry(self, plot_joints=True):
        # Export Geometry for Visualization
        geometry = self.meshs.copy()
        if plot_joints:
            for joints in self.joints:
                joints_pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(joints))
                joints_pcl.paint_uniform_color([1.0, 0.0, 0.0])
                geometry.append(joints_pcl)
        return geometry
        # import open3d.web_visualizer
        # open3d.web_visualizer.draw(geometry)

    def export_mesh_to_file(self, filename):
        o3d.io.write_triangle_mesh(str(filename), self.meshs[0], write_triangle_uvs=False)


class CanonicalScene(_HumanScene):
    def __init__(self, scene, cfg, model_type='none', **kwargs) -> None:
        super().__init__()

        PoseSkeleton = MyPoseSkeleton(cfg=cfg)
        self.PoseSkeleton = PoseSkeleton
        self.vertices, self.joints, self.keypoints, self.triangles = self.PoseSkeleton()
        
        # Build Scene
        self.meshs, self.ray_casting_scene = self.build_scene()

    def set_frame_index(self, frame_idx):
        pass

class PoseSkeletonPrompt:
    def __init__(self, cfg: PromptConfig, cond_type, scene='canonical-A', num_person=1, height=512, width=512):
        # Init
        self.cfg = cfg
        self.cond_type = cond_type
        self.height, self.width = height, width

        self.hs = CanonicalScene(scene=scene, cfg=cfg)
        

    def __call__(self, intrinsics, cam2world, dirs, cond_type=None, frame_idx=None):
        """
        Input:
            intrinsics: shape = [4, ]
            cam2world: shape = [N, 4, 4]
            cond_type: List[str]
        Return:
            cond_images: list of [PIL.Image]
        """
        if cond_type is None:
            cond_type = self.cond_type
        if isinstance(cond_type, str):
            cond_type = [cond_type,]
        intrinsics, extrinsic = NeRF_data_to_standard(intrinsics, cam2world, H=self.height, W=self.width)
        cond_images = []
        for _cond in cond_type:
            if _cond == 'pose':
                cond_image = self.hs.export_pose_map(intrinsics, extrinsic, dirs, width=self.width, height=self.height, occlusion_culling=self.cfg.occlusion_culling)
            elif _cond == 'depth':
                cond_image = self.hs.export_depth_map(intrinsics, extrinsic, width=self.width, height=self.height)
            elif _cond == 'mesh':
                cond_image = self.hs.export_mesh_map(intrinsics, extrinsic, width=self.width, height=self.height)
            else:
                assert 0, _cond
            cond_images.append(cond_image)
        return cond_images

    def write_video(self, view_prompt, save_dir='./', save_image='output.png', save_video='output.mp4', cond_type=None):
        import os
        import os.path as osp
        os.makedirs(save_dir, exist_ok=True)
        images = []
        for i in range(100):
            intrinsics, cam2world, dirs = index2pose(i, view_prompt, H=self.height, W=self.width)
            image = self(intrinsics, cam2world, dirs, cond_type=cond_type)[0]
            if i == 0:
                image.save(osp.join(save_dir, save_image))
            images.append(np.array(image))
        imageio.mimsave(osp.join(save_dir, save_video), np.array(images), fps=25, quality=8, macro_block_size=1)
        return image
