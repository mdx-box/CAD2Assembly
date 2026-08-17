# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy

import numpy as np
import nvdiffrast.torch as dr

from foundationpose.Utils import *
from foundationpose.datareader import *
import itertools
from foundationpose.learning.training.predict_score import *
from foundationpose.learning.training.predict_pose_refine import *

class MultiView:
    def __init__(self, symmetry_tfs=None, mesh=None, scorer: ScorePredictor = None,
                 refiner: PoseRefinePredictor = None, glctx=None, debug=0,
                 debug_dir='./debug/'):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir

        self.refiner_dir = os.path.join(self.debug_dir,"refine")
        self.scorer_dir = os.path.join(self.debug_dir,"score")
        if self.debug != 0:
            os.makedirs(debug_dir, exist_ok=True)
            os.makedirs(self.refiner_dir, exist_ok=True)
            os.makedirs(self.scorer_dir, exist_ok=True)

        # self.reset_object(mesh=mesh, symmetry_tfs=symmetry_tfs)
        # self.make_rotation_grid(min_n_views=40, inplane_step=60)

        # self.glctx = glctx
        self.reset_object(mesh=mesh, symmetry_tfs=symmetry_tfs)
        #min_n_views42个观察方向，inplane_step60度旋转步长，生成了252个初始旋转假设
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        max_pose_hypotheses = int(os.environ.get("MULTIVIEW_MAX_POSE_HYPOTHESES", "252"))

        if max_pose_hypotheses > 0 and len(self.rot_grid) > max_pose_hypotheses:
            print(f"[MultiView] Limit rot_grid: {len(self.rot_grid)} -> {max_pose_hypotheses}")
            self.rot_grid = self.rot_grid[:max_pose_hypotheses]
        #glctx是nvdiffrast的渲染上下文，默认为CudaContext，可以选择RasterizeGLContext
        self.glctx = glctx

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = ScorePredictor()

        if refiner is not None:
            self.refiner = refiner
        else:
            self.refiner = PoseRefinePredictor()

        self.pose_last = None  # Used for tracking; per the centered mesh

    #将输入的 Mesh 转换成 FoundationPose 能够用于渲染、评分和位姿优化的内部表示
    def reset_object(self, mesh=None, symmetry_tfs=None):


        # center = mesh_o3d.get_oriented_bounding_box(robust=True).center
        # self.model_center = center
        #get the min and max xyz of the mesh vertices to compute the center of the model, and then translate the mesh to be centered at the origin
        min_xyz = mesh.vertices.min(axis=0)
        max_xyz = mesh.vertices.max(axis=0)
        self.model_center = (min_xyz+max_xyz)/2
        #MultiView 后续的位姿估计、render-and-compare、mesh tensor 构造，都默认物体模型是在一个统一的局部坐标系下处理的。
        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)
        #这几行将 trimesh.Trimesh 格式转换为 open3d.geometry.TriangleMesh 格式
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        #如果 mesh 自带 vertex colors，则转成 Open3D 颜色
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            rgb_colors = mesh.visual.vertex_colors[:, :3].astype(float) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(rgb_colors)
        #如果 mesh 有 UV 纹理，则从纹理图中采样顶点颜色
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and mesh.visual.material.image is not None:
           #uv 通常表示每个顶点在纹理图上的二维坐标
           #把纹理图像转成 RGB 格式，再转成 NumPy array, (H,W,3)
            img = copy.deepcopy(np.array(mesh.visual.material.image.convert('RGB')))

            uv = copy.deepcopy(mesh.visual.uv)
            #uv.shape = (N, 2)，范围是[0,1] N = 顶点数（vertex number）
            #不同图形系统对纹理坐标原点定义不同,UV 坐标中，v=0 常常表示纹理底部,图像数组中，row=0 表示图像顶部
            uv[:, 1] = 1 - uv[:, 1]
            #uv to pixel coordinates
            uv_pixels_y = np.clip((uv[:, 1] * img.shape[0]).astype(int), 0, img.shape[0] - 1)
            uv_pixels_x = np.clip((uv[:, 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1)

            vertex_colors = img[uv_pixels_y, uv_pixels_x].astype(float) / 255.0

            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        #计算 Open3D mesh 的顶点法向量
        mesh_o3d.compute_vertex_normals()
        self.mesh_o3d = mesh_o3d
        # model_pts = mesh.vertices
        #计算物体直径和体素尺度,n_sample=10000 表示可能采样 10000 个点来估计直径，而不是对所有顶点做完整两两距离计算
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        #如果物体比较大，体素大小约为：直径的 1/20；如果物体比较小，体素大小至少为 3mm。这是为了在后续的点云处理和渲染中，保持一个合理的分辨率和平滑度。
        self.vox_size = max(self.diameter / 20.0, 0.003)
        # logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg

        self.mesh = mesh
        #把 mesh 转成神经网络和 differentiable rendering 能用的 tensor 格式
        #self.mesh_tensors 是 render-and-compare 过程中真正供网络和渲染器使用的 mesh 表示
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

        # logging.info("reset done")
    #构造一个 4×4 齐次变换矩阵，用来把原始 mesh 坐标系中的点转换到“中心化 mesh 坐标系”
    def get_tf_to_centered_mesh(self):
        #tf_to_center 是一个 4×4 的齐次变换矩阵，表示从原始 mesh 坐标系到中心化 mesh 坐标系的变换, R(3x3) and T(3x1)
        tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
        #把点从原始 mesh 坐标系平移到中心化 mesh 坐标系, 平移向量是 -self.model_center, 因为中心化坐标系的原点在模型中心，所以需要把模型坐标系中的点沿 x,y,z 轴分别平移 -model_center.x, -model_center.y, -model_center.z
        tf_to_center[:3, 3] = -torch.FloatTensor(self.model_center.copy()).cuda()
        return tf_to_center

    def to_device(self, s='cuda:0'):
        #遍历当前 MultiView 对象的所有属性名。
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k]
            if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
                # logging.info(f"Moving {k} to device {s}")
                self.__dict__[k] = self.__dict__[k].to(s)
        for k in self.mesh_tensors:
            # logging.info(f"Moving {k} to device {s}")
            self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        if self.refiner is not None:
            self.refiner.model.to(s)
        if self.scorer is not None:
            self.scorer.model.to(s)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(s)
    #预先生成一组离散的候选旋转姿态，作为后续 6D 位姿估计的初始 pose hypotheses。
    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        '''min_n_views: 最少采样多少个观察方向 inplane_step: 表示每个观察方向下，绕相机视线方向进行多少度一次的平面内旋转'''
        #在球面上采样若干个相机观察方向,假设物体在原点周围，相机从不同方向观察物体，那么每一个方向都对应一个可能的物体朝向
        #cam_in_obs.shape = (N, 4, 4)，N 是采样的观察方向数量，每个观察方向对应一个从相机坐标系到物体坐标系的变换矩阵
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        # logging.info(f'cam_in_obs:{cam_in_obs.shape}')
        #创建空的旋转候选列表
        rot_grid = []
        #遍历所有观察方向
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                #取出第 i 个观察方向
                cam_in_ob = cam_in_obs[i]
                #生成一个欧拉角旋转矩阵, c构造绕 z 轴的平面内旋转矩阵,绕 x 轴旋转 0度，绕 y 轴旋转 0 度，绕 z 轴旋转 inplane_rot 度
                R_inplane = euler_matrix(0, 0, inplane_rot)
                #把原始观察方向和绕 z 轴的平面内旋转组合起来,完整候选相机姿态 = 球面观察方向 × 平面内旋转
                cam_in_ob = cam_in_ob @ R_inplane
                #cam_in_ob表示相机坐标系在物体坐标系下的位姿,这里取逆的目的就是把视角采样得到的相机姿态转换成物体姿态假设
                ob_in_cam = np.linalg.inv(cam_in_ob)
                #把当前生成的候选姿态加入列表
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        # logging.info(f"rot_grid:{rot_grid.shape}")
        #调用 C++ 扩展 mycpp.cluster_poses() 对姿态候选进行聚类/去重,把过密、重复、对称等价的旋转候选合并
        rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        # 转成 CUDA tensor 保存sample_views_icosphere
        # logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
        self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
        # logging.info(f"self.rot_grid: {self.rot_grid.shape}")
        #把前面 make_rotation_grid() 生成的一组旋转候选，加上一个根据 depth 和 mask 粗略估计出来的平移中心，形成完整的初始 6D pose hypotheses
    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None,initial_center=False):
        '''
        @scene_pts: torch tensor (N,3)
        '''
        #从 self.rot_grid 复制一份候选姿态
        ob_in_cams = self.rot_grid.clone()
        if initial_center:

            center = self.guess_translation_bounding_box(depth=depth, mask=mask, K=K)
        else:
            #目标物体中心在相机坐标系下的粗略三维位置
            center = self.guess_translation(depth=depth, mask=mask, K=K)
        #把中心平移写入所有候选姿态
        ob_in_cams[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1, 3)
        #返回完整的初始 pose hypotheses
        return ob_in_cams
    #根据 depth、mask 和相机内参 K，粗略估计目标物体在相机坐标系下的三维中心位置,作为所有初始 pose hypotheses 的平移部分
    def guess_translation_bounding_box(self, depth, mask, K):
        #  用 mask 区域 depth 反投影成 3D 点云，再取 3D OBB 中心
        #将 depth 图反投影成 3D 点云图
        xyz_map = depth2xyzmap(depth, K) #xyz_map.shape = (H,W,3)，每个像素位置对应一个三维坐标
        #去掉非目标区域的 3D 点
        xyz_map[mask == False] = 0
        #从 xyz_map 中取出所有 mask=True 的位置
        points = xyz_map[mask].reshape(-1, 3)
        #创建一个 Open3D 点云对象。
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        #去除统计离群点,对每个点，计算它到邻近点的平均距离。如果某些点和周围点距离明显偏大，就认为它们是离群点，将其剔除。
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(np.array(points).shape[0] * 0.01), std_ratio=2.0)
        #估计点云法向量
        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
        #让点云法向量方向在局部切平面上保持一致。
        pcd_clean.orient_normals_consistent_tangent_plane(k=1000)
        #计算目标点云的 OBB，即 oriented bounding box，有向包围盒
        obb_pcd_clean = pcd_clean.get_oriented_bounding_box()
        # 取 OBB 中心作为平移估计
        center = obb_pcd_clean.get_center()

        return np.asarray(center)

    def guess_translation(self, depth, mask, K):
        #找出所有属于目标物体的像素位置
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            #mask 里没有任何目标像素,这种情况下无法估计物体中心,只能返回一个默认值（比如全零），并且在日志中记录这个异常情况
            # logging.info(f'mask is all zero')
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        #找到 mask 内有效深度区域
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            #如果没有有效深度，返回零向量
            # logging.info(f"valid is empty")
            return np.zeros((3))
        #用深度中位数作为目标中心深度,相比平均值，中位数对异常值更鲁棒
        zc = np.median(depth[valid])
        #将 2D bbox 中心反投影成 3D center
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc


        return center.reshape(3)

    def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, name=None,no_center=False, initial_center=False):
        '''Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        #给定 RGB 图像、深度图、目标 mask、相机内参和物体 mesh，估计该物体在相机坐标系下的 6D pose
        输入预处理 → 初始姿态生成 → refiner 优化 → scorer 打分 → 选最优位姿 → 输出坐标系修正
        register() 更接近 FoundationPose 原始注册逻辑
        '''
        set_seed(0)
        # logging.info('Welcome')
        #初始化 nvdiffrast 渲染上下文
        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
                # self.glctx = dr.RasterizeGLContext()
            else:
                self.glctx = glctx
        #对深度图做腐蚀操作，通常用于处理目标边缘噪声
        depth = erode_depth(depth, radius=2, device='cuda')
        #双边滤波用于平滑深度图，同时尽量保留边缘
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        #把目标 mask 之外的深度全部置零
        depth[ob_mask==False] = 0

        #告诉网络当前没有额外法向量信息
        normal_map = None
        #
        valid = (depth >= 0.001) & (ob_mask > 0)
        #如果深度点太少，系统放弃 refiner/scorer，只返回一个粗略平移结果
        if valid.sum() < 4:
            # logging.info(f'valid too small, return')
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose
        #保存当前图像尺寸和输入信息
        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask
        #生成初始 pose hypotheses
        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None,initial_center=initial_center)
        poses = poses.data.cpu().numpy()
        # logging.info(f'poses:{poses.shape}')随后代码又重新计算一次中心
        if initial_center:
            center = self.guess_translation_bounding_box(depth=depth, mask=ob_mask, K=K)
        else:
            center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
        #转回 CUDA tensor
        poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')
        #计算 ADD 误差
        add_errs = self.compute_add_err_to_gt_pose(poses)
        # logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")
        #将深度图转换为 xyz_map
        xyz_map = depth2xyzmap(depth, K)
        #使用 refiner 优化候选 pose,对每个初始 pose hypothesis，进行若干轮迭代修正，使渲染出来的物体与真实 RGB-D 观测更对齐
        poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                          glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.refiner_dir}/vis_refiner.png', vis)
        #使用 scorer 对优化后的 pose 打分,哪个 pose 渲染出来后，最符合当前 RGB-D 观测？
        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.scorer_dir}/vis_score.png', vis)

        add_errs = self.compute_add_err_to_gt_pose(poses)
        # logging.info(f"final, add_errs min:{add_errs.min()}")
        # 根据得分从高到低排序
        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]

        # logging.info(f'sorted scores:{scores}')
        # 得到最终 best pose，并做中心化坐标修正,也就是相对于原始 mesh 坐标系的最终 pose
        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        #保存跟踪状态和候选结果
        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores
        if no_center:
            return poses[0].data.cpu().numpy()
        else:
            return best_pose.data.cpu().numpy()


    def register_multiview(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, name=None, refinement=True, axis_align=True,coarse_est=True):
        '''Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        点云/尺度粗对齐、第一轮 pose refinement/scoring、基于最佳 pose 的尺度再估计、最终重跑 pose refinement/scoring
        register_multiview():
        深度点云与 mesh 粗尺度对齐
            ↓
        第一轮 pose refinement / scoring
            ↓
        根据最佳 pose 把观测点云变换到物体坐标系
            ↓
        再次估计 mesh 尺度
            ↓
        随机采样多组 scale，refiner/scorer 选择最佳 scale
            ↓
        用修正后的 mesh 重新 refine pose
            ↓
        返回最终 pose
        '''
        set_seed(0)

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = glctx
        #对深度图做腐蚀和双边滤波，然后将深度图反投影为三维点云图
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        xyz_map = depth2xyzmap(depth, K)
        #只保留 mask 内目标点云,
        # 从 RGB-D 图像中提取当前观测到的目标物体点云，用它来和当前 mesh 做尺度/方向对齐。
        xyz_map[ob_mask == False] = 0
        points = xyz_map[ob_mask].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0.529, 0.808, 0.922], (len(pcd.points), 1)))  # 모든 point를 파란색으로

        #去除点云离群点，并把点云移动到自身 OBB 中心附近
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(points.shape[0] * 0.01), std_ratio=2.0)
        #把观测点云中心移动到原点
        pcd_clean.translate(-pcd_clean.get_oriented_bounding_box(robust=True).center)
        #估计点云法向量并计算观测点云 OBB
        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
        pcd_clean.orient_normals_consistent_tangent_plane(k=1000)
        obb_pcd_clean = pcd_clean.get_oriented_bounding_box()
        obb_pcd_clean.color = (0, 0, 0)
        #如果 coarse_est=True，先进行 mesh 和观测点云的粗尺度对齐
        if coarse_est:
            #从当前 mesh 上均匀采样 100000 个点，然后计算 mesh 点云的 OBB
            mesh_pcd = copy.deepcopy(self.mesh_o3d)
            pcd_ = mesh_pcd.sample_points_uniformly(number_of_points=100000)
            #去除 mesh 点云离群点，并计算 mesh 点云的 OBB
            cl, _ = pcd_.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # oriented_bounding_box() 计算 mesh 点云的 OBB，得到 mesh 的主轴方向和尺度信息
            obb_mesh = cl.get_oriented_bounding_box()
            obb_mesh.color = (0, 0, 1)
            #对齐观测点云和 mesh 的 OBB 主轴,观测点云的长宽高方向和mesh 的长宽高方向
            if axis_align:
                pcd_clean.rotate(obb_mesh.R @ obb_pcd_clean.R.T, center=obb_pcd_clean.center)
            else:
                pass
            obb_pcd_clean = pcd_clean.get_oriented_bounding_box(robust=True)
            obb_pcd_clean.color = (0, 1, 0)

            #根据 OBB 尺寸估计粗尺度比例
            extent_pcd_clean = obb_pcd_clean.extent #观测点云 OBB 的长宽高
            extent_mesh = obb_mesh.extent #mesh OBB 的长宽高
            #找一个较合适的尺度比例。最后对 mesh 点云做缩放
            ratio, best_perm, best_iou = find_best_ratio_combination(extent_pcd_clean, extent_mesh, obb_pcd_clean, obb_mesh)
            #x/y/z 同时乘以同一个比例,也就是说，这一阶段粗略地选择了 y 方向的比例作为整体缩放比例
            mesh_pcd.scale(ratio[1], center=obb_mesh.center) # y axis (height)
            #用缩放后的 mesh 更新OBB
            obb_mesh = mesh_pcd.get_oriented_bounding_box(robust=True)
            obb_mesh.color = (1, 0, 0)
            #把缩放后的 Open3D mesh 顶点写回到 trimesh mesh 中，然后调用 reset_object() 重新初始化内部物体表示
            mesh = copy.deepcopy(self.mesh)
            mesh.vertices = np.asarray(mesh_pcd.vertices)
            self.reset_object(mesh=mesh, symmetry_tfs=self.symmetry_tfs)
            self.mesh.export(os.path.join(self.debug_dir, f'refine_init_mesh_{name}.obj'))
        #利用前面 remove_statistical_outlier() 返回的 ind，把去噪点云对应回原图像像素位置，然后构建一个新的 depth_mask
        valid_indices = np.argwhere(ob_mask)  # Get (h, w) coordinates where ob_mask is True
        selected_indices = valid_indices[ind]  # Use the indices from the outlier filtering to get (h, w)
        #depth重新构建
        depth_mask = np.zeros((xyz_map.shape[:2]), dtype=bool)
        depth_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
        xyz_map[depth_mask == False] = 0

        depth = xyz_map[..., -1]

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            # logging.info(f'valid too small, return')
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask
        #生成第一轮初始 pose hypotheses
        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
        poses = poses.data.cpu().numpy()
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')

        xyz_map = depth2xyzmap(depth, K)
        #第一轮 pose refinement
        poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                          glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                          get_vis=self.debug >= 2)

        if vis is not None:
            imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_1_consider_pose_{name}.png', vis)
        #第一轮 refined poses 进行评分
        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                          get_vis=self.debug >= 2)
        if vis is not None:
            imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_1_consider_pose_{name}.png', vis)

        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]
        best_pose = poses[0].data.cpu().numpy()
        #best_pose is T_cam_obj,物体坐标系到相机坐标系的变换



        if refinement:#
            #根据第一轮最佳 pose，把观测点云变换到物体坐标系下，进行尺度再估计和第二轮 pose refinement
            cam_in_ob = np.linalg.inv(best_pose)
            #mask to depth
            points = xyz_map[ob_mask].reshape(-1, 3)
            #pcd object
            pcd = o3d.geometry.PointCloud()
            #points to pcd 格式转换
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([1,0,0])
            #去除点云离群点，并把点云移动到自身 OBB 中心附近
            pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(points.shape[0] * 0.01), std_ratio=2.0)
            #把相机坐标系下的观测点云变换到物体坐标系下
            pcd_clean.transform(cam_in_ob)
            #计算 mesh OBB 和变换后观测点云 OBB
            mesh_pcd = (copy.deepcopy(self.mesh_o3d))
            obb_mesh = mesh_pcd.get_oriented_bounding_box(robust=True)
            obb_mesh.color = (0, 1, 0)
            #mesh OBB 是“模型自身”的包围盒；变换后观测点云 OBB 是“query 深度点云被拉回到物体坐标系后”的包围盒。二者都应该在物体坐标系附近进行比较
            obb_pcd_clean = pcd_clean.get_oriented_bounding_box(robust=True)
            obb_pcd_clean.color = (0, 1, 0)
            #把观测点云 OBB 的主轴对齐到 mesh OBB 的主轴
            if axis_align:
                pcd_clean.rotate(obb_mesh.R @ obb_pcd_clean.R.T, center=obb_pcd_clean.center)
            else:
                pass
            obb_pcd_clean = pcd_clean.get_oriented_bounding_box(robust=True)
            obb_pcd_clean.color = (1, 0, 0)

            # o3d.visualization.draw_geometries([obb_mesh, pcd_clean, mesh_pcd,coordinate_frame, obb_pcd_clean])
            #根据 OBB extent 做三轴尺度修正
            extent_pcd_clean = obb_pcd_clean.extent  #变换后观测点云 OBB 的长宽高
            extent_mesh = obb_mesh.extent     #mesh OBB 的长宽高
            ratio = extent_pcd_clean / extent_mesh #ratio 是一个三维向量，表示 x/y/z 轴的尺度修正比例
            mesh_pcd.translate(obb_pcd_clean.center - obb_mesh.center)   #先把 mesh 点云平移到观测点云 OBB 中心位置，这样后续的缩放才是以观测点云 OBB 中心为基准的
            #对 mesh 顶点做三轴非均匀缩放
            mesh_pcd.vertices = o3d.utility.Vector3dVector(np.array(mesh_pcd.vertices) * ratio[None])
            mesh_pcd.translate(obb_mesh.center - obb_pcd_clean.center) #再把 mesh 点云平移回原来位置，这样就完成了以观测点云 OBB 中心为基准的非均匀缩放
            #更新 mesh OBB
            obb_mesh = mesh_pcd.get_oriented_bounding_box(robust=True)
            obb_mesh.color = (0, 1, 0)
            # o3d.visualization.draw_geometries([obb_mesh, pcd_clean, mesh_pcd,coordinate_frame, obb_pcd_clean])
            #用修正后的 mesh 再次 reset_object, 重新计算model_center, mesh_o3d, diameter, vox_size, mesh_tensors
            mesh = copy.deepcopy(self.mesh)
            mesh.vertices = np.asarray(mesh_pcd.vertices)
            self.reset_object(mesh=mesh, symmetry_tfs=self.symmetry_tfs)
            #第二轮 pose refinement / scoring 用经过三轴 OBB 尺度修正后的 mesh，再重新估计一次 pose。
            poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
            poses = poses.data.cpu().numpy()
            poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
            poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')


            poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                              glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                              get_vis=self.debug >= 2)

            if vis is not None:
                imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_2_consider_pose_{name}.png', vis)

            scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                              mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                              get_vis=self.debug >= 2)
            if vis is not None:
                imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_2_consider_pose_{name}.png', vis)

            ids = torch.as_tensor(scores).argsort(descending=True)
            # logging.info(f'sort ids:{ids}')
            scores = scores[ids]
            poses = poses[ids]
            # endregion


        #随机采样 252 组三轴 scale
        if refinement:
            best_pose = poses[0].data.cpu().numpy()
            if True:
                # 252 samples
                # Define the number of samples
                num_samples = int(os.environ.get("MULTIVIEW_RESCALE_SAMPLES", "32"))

                # Define the scaling ratios for each axis 为 x、y、z 三个方向分别随机采样 252 个缩放因子
                ratio_i, ratio_l = 0.6, 1.4
                ratios = {'x': (ratio_i, ratio_l), 'y': (ratio_i, ratio_l), 'z': (ratio_i, ratio_l)}

                # Generate random scaling values for each axis
                samples = {axis: np.random.uniform(*ratio, num_samples) for axis, ratio in ratios.items()}
                # This creates a dictionary with keys 'x', 'y', 'z', each containing an array of 252 random values

                # Create scaling matrices 构造 252 个 scale matrix，并组合到 best_pose 上
                scaling_matrices = np.array([np.diag([samples['x'][i], samples['y'][i], samples['z'][i], 1]) for i in range(num_samples)])
                # This creates 252 4x4 diagonal matrices, each representing a scaling transformation

                # Generate final transformation matrices, best_pose @ scaling_matrices
                final_transforms = np.einsum('ij,njk->nik', best_pose, scaling_matrices)
                # This applies the best_pose transformation to each scaling matrix [252, 4, 4]
                #用 refiner/scorer 评估这些 scale candidates
                rescale_poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb,
                                                          depth=depth,
                                                          K=K,
                                                          ob_in_cams=final_transforms, normal_map=normal_map,
                                                          xyz_map=xyz_map,
                                                          glctx=self.glctx, mesh_diameter=self.diameter,
                                                          iteration=iteration,
                                                          get_vis=self.debug >= 2)
                if vis is not None:
                    imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_3_consider_size.png', vis)

                rescale_scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                                          ob_in_cams=rescale_poses.data.cpu().numpy(),
                                                          normal_map=normal_map,
                                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                                          mesh_diameter=self.diameter,
                                                          get_vis=self.debug >= 2)
                if vis is not None:
                    imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_3_consider_size.png', vis)
                print('')

                # combine_scores = scores + rescale_scores
                # combine_ids = torch.as_tensor(combine_scores).argsort(descending=True)
                rescale_ids = torch.as_tensor(rescale_scores).argsort(descending=True)

                # rescale_scores = rescale_scores[rescale_ids]
                # rescale_poses = rescale_poses[rescale_ids]
                scaling_matrices = scaling_matrices[rescale_ids.detach().cpu().numpy()]
                #选出最优的 scale matrix
                scale = np.array([scaling_matrices[0][0, 0], scaling_matrices[0][1, 1], scaling_matrices[0][2, 2]])
                print(f"scale {scale}")
                #对 mesh 顶点做三轴缩放
                self.mesh.vertices = self.mesh.vertices * scale

            #用最终 scale 后的 mesh 重新初始化，并导出 final mesh
            self.reset_object(mesh=self.mesh, symmetry_tfs=self.symmetry_tfs)
            self.mesh.export(os.path.join(self.debug_dir, f'final_mesh_{name}.obj'))
            #用最终 mesh 重跑 pose refinement / scoring
            poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                              xyz_map=xyz_map,
                                              glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                              get_vis=self.debug >= 2)
            if vis is not None:
                imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_4_rerun_pose.png', vis)

            scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                              mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                              mesh_diameter=self.diameter,
                                              get_vis=self.debug >= 2)
            if vis is not None:
                imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_4_rerun_pose.png', vis)

            ids = torch.as_tensor(scores).argsort(descending=True)
            # logging.info(f'sort ids:{ids}')
            scores = scores[ids]
            poses = poses[ids]
        # logging.info(f'sorted scores:{scores}')
        #输出最终 pose，并保存内部状态
        best_pose = poses[0] @ self.get_tf_to_centered_mesh()


        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores
        return best_pose.data.cpu().numpy()
    '''
    Stage 0: 输入预处理
        depth erode + bilateral filter
        depth → xyz_map
        mask 提取目标点云
        去除离群点

    Stage 1: 粗尺度估计 + 第一轮 pose
        mesh 点云 OBB
        观测点云 OBB
        OBB 主轴对齐
        粗略缩放 mesh
        reset_object()
        生成 pose hypotheses
        refiner + scorer
        得到 first best pose

    Stage 2: 基于 first best pose 的尺度修正
        把观测点云从相机坐标系变换到物体坐标系
        再次比较 mesh OBB 和点云 OBB
        三轴 scale 修正 mesh
        reset_object()
        第二轮 refiner + scorer

    Stage 3: 随机 scale search
        采样 252 组三轴 scale
        best_pose × scale_matrix
        refiner + scorer 选择最佳 scale
        缩放 mesh
        reset_object()
        导出 final mesh

    Stage 4: 最终 pose refinement
        用最终 mesh 重跑 refiner + scorer
        选择最高分 pose
        坐标系修正
        返回最终 pose
    '''

    def compute_add_err_to_gt_pose(self, poses):
        '''
        @poses: wrt. the centered mesh
        '''
        return -torch.ones(len(poses), device='cuda', dtype=torch.float)

    def track_one(self, rgb, depth, K, iteration, extra=None, no_center=False):
        if extra is None:
            extra = {}
        #在已经有上一帧物体位姿的情况下，用当前帧 RGB-D 图像对上一帧 pose 做局部 refinement，从而得到当前帧 pose
        if self.pose_last is None:
            # logging.info("Please init pose by register first")
            raise RuntimeError
        # logging.info("Welcome")

        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        # logging.info("depth processing done")
        #相机内参也加 batch 维度
        xyz_map = \
        depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]
        #
        pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                         ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(), normal_map=None,
                                         xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx,
                                         iteration=iteration, get_vis=self.debug >= 2)
        # logging.info("pose done")
        if self.debug >= 2:
            extra['vis'] = vis
        self.pose_last = pose
        if no_center:
            return pose[0].data.cpu().numpy()
        else:
            #把中心化 mesh 的 pose 转换回原始 mesh 坐标系下的 pose
            return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)

    def track_one_multiview(self, rgb, depth, K, iteration, extra=None):
        if extra is None:
            extra = {}
        if self.pose_last is None:
            # logging.info("Please init pose by register first")
            raise RuntimeError
        # logging.info("Welcome")

        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        # logging.info("depth processing done")

        xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

        pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                         ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(), normal_map=None,
                                         xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx,
                                         iteration=iteration, get_vis=self.debug >= 2)
        # logging.info("pose done")
        if self.debug >= 2:
            extra['vis'] = vis
        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)
