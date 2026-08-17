import os
import sys
import math
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from torch.nn.functional import cosine_similarity
from sklearn.neighbors import KDTree
from typing import Tuple
from torch import Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.distributions import Categorical
from bop_toolkit.bop_toolkit_lib.pose_error_custom import my_mspd, my_mssd, vsd


def unique_matches(matches: torch.Tensor) -> torch.Tensor:
    '''
    Given a set of 2D-2D matches (Nx4), returns only unique matches
    '''

    assert matches.max() < 999, ' cannot handle indexes > 999'
    matches = matches.to(torch.int)
    match_set = set()
    for match in matches:
        match_code = f'{match[0]:03d}-{match[1]:03d}-{match[2]:03d}-{match[3]:03d}'
        match_set.add(match_code)

    match_list = list()
    for match in match_set:
        match_idxs = [int(m) for m in match.split('-')]
        match_list.append(match_idxs)

    return torch.tensor(match_list).to(torch.float)


def torch_sample_select(t : torch.Tensor, n : int) -> torch.Tensor:
    '''
    Samples exactly n elements from a Tensor of shape [N,D1,....,Dm]
    Uses replacement only if n > N.
    Returns indexes
    '''

    N = t.shape[0]
    uniform_dist = torch.ones(N,dtype=float).to(t.device)
    if n > N:
        return torch.multinomial(uniform_dist, n, replacement=True).to(t.device)
    else:
        return torch.multinomial(uniform_dist, n, replacement=False).to(t.device)

def get_diameter(pcd : np.ndarray) -> float:

    xyz = pcd[:,:3]
    maxs, mins = np.max(xyz,axis=0), np.min(xyz,axis=0)
    return max(maxs-mins)


def format_sym_set(syms: dict) -> np.ndarray:
  '''
  Format a symmetry set provided by BOP into a nd array od shape [N,3,4]
  '''

  syms_r = np.stack([np.asarray(sym['R']) for sym in syms],axis=0)
  syms_t = np.stack([np.asarray(sym['t']) for sym in syms],axis=0)
  sym_poses = np.concatenate([syms_r,syms_t],axis=2)

  return sym_poses


def np_transform_pcd(pcd: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    pcd = pcd.astype(np.float16)
    r = r.astype(np.float16)
    t = t.astype(np.float16)
    rot_pcd = np.dot(np.asarray(pcd), r.T) + t
    return rot_pcd



def project_points(v: np.ndarray, k: np.ndarray) -> np.ndarray:
    if len(v.shape) == 1:
        v = np.expand_dims(v, 0)

    assert len(v.shape) == 2, '  wrong dimension, expexted shape 2.'
    assert v.shape[1] == 3, ' expected 3d points, got ' + str(v.shape[0]) + ' ' + str(v.shape[1]) + 'd points instead.'

    k = k.astype(np.float16)
    p = np.matmul(k, v.T)
    p[0] = p[0] / (p[2] + 1e-6)
    p[1] = p[1] / (p[2] + 1e-6)

    return p[:2].T



def nn_correspondences(feats1: Tensor, feats2: Tensor, mask1: Tensor, mask2: Tensor,
                       threshold: float, max_corrs: int, subsample_source: int = None,
                       corrs_device: str = 'cuda') -> Tensor:
    """
    Finds matches between two [D,H,W] feature maps with robust error handling.

    Args:
        feats1: First feature map [D,H,W]
        feats2: Second feature map [D,H,W]
        mask1: First mask [H,W]
        mask2: Second mask [H,W]
        threshold: Distance threshold for valid correspondences
        max_corrs: Maximum number of correspondences to return
        subsample_source: Number of source points to sample (optional)
        corrs_device: Device to compute correspondences on ('cuda' or 'cpu')

    Returns:
        Tensor: Correspondences in shape (N,4) format (y1,x1,y2,x2) or None if no valid matches
    """
    # Input validation
    if not (feats1.dim() == 3 and feats2.dim() == 3):
        raise ValueError(f"Features must be 3D (D,H,W), got shapes {feats1.shape}, {feats2.shape}")
    if not (mask1.dim() == 2 and mask2.dim() == 2):
        raise ValueError(f"Masks must be 2D (H,W), got shapes {mask1.shape}, {mask2.shape}")

    # Store original device for final output
    orig_device = feats1.device

    # Find valid points in masks
    roi1 = torch.nonzero(mask1 == 1)
    roi2 = torch.nonzero(mask2 == 1)

    # Check if we have enough points to match
    if roi1.shape[0] == 0 or roi2.shape[0] == 0:
        # warnings.warn("No valid points found in one or both masks")
        return None

    # Move to specified device
    roi1 = roi1.to(corrs_device)
    roi2 = roi2.to(corrs_device)

    # Subsample source points if requested
    if subsample_source is not None and roi1.shape[0] > subsample_source:
        idxs = torch_sample_select(roi1, subsample_source)
        roi1 = roi1[idxs]

    # Extract features for valid points
    try:
        roi_feats1 = feats1[:, roi1[:, 0], roi1[:, 1]].T.to(corrs_device)
        roi_feats2 = feats2[:, roi2[:, 0], roi2[:, 1]].T.to(corrs_device)
    except IndexError as e:
        warnings.warn(f"Failed to extract features: {str(e)}")
        return None

    # Convert to appropriate precision based on device
    if corrs_device == 'cuda':
        roi_feats1 = roi_feats1.to(torch.float16).cuda()
        roi_feats2 = roi_feats2.to(torch.float16).cuda()
    else:
        roi_feats1 = roi_feats1.to(torch.float32).cpu()
        roi_feats2 = roi_feats2.to(torch.float32).cpu()

    # Compute pairwise distances
    try:
        dist = pdist(roi_feats1, roi_feats2, 'inv_norm_cosine')
    except RuntimeError as e:
        warnings.warn(f"Failed to compute distances: {str(e)}")
        return None

    # Find nearest neighbors
    min_dist = torch.amin(dist, dim=1)
    roi2_idxs = torch.argmin(dist, dim=1)

    # Filter valid correspondences
    valid_corr = torch.nonzero(min_dist < threshold)

    if valid_corr.shape[0] <= 1:
        warnings.warn("No valid correspondences found within threshold")
        return None

    # Get matched points
    roi2_matched = roi2[roi2_idxs]
    final_corrs = torch.cat((roi1[valid_corr.squeeze(1)],
                             roi2_matched[valid_corr.squeeze(1)]), dim=1)

    # Sample if we have more correspondences than requested
    if final_corrs.shape[0] > max_corrs:
        idxs = torch_sample_select(final_corrs, max_corrs)
        final_corrs = final_corrs[idxs]

    # Return to original device
    return final_corrs.to(orig_device)
def compute_pose_metrics_ho3d(metrics, pred_q, gt_pose_q, obj_model, obj_diam, obj_symms, depth_q, K, renderer, obj_f, instance_id):
    """
    Compute various pose estimation metrics including R/T errors, ADD(S), MSSD, MSPD, and VSD.

    Args:
        pred_q: Predicted pose matrix
        gt_pose_q: Ground truth pose matrix
        obj_model: Object model dictionary containing 'pts' key
        obj_diam: Object diameter
        obj_symms: Object symmetries
        depth_q: Depth image
        K: Camera intrinsic matrix
        renderer: Renderer object for VSD computation
        obj_f: Object ID/filename
        instance_id: Instance ID

    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Add pose recall metrics dynamically
    pose_recall_th = [(5, 5), (10, 10), (15, 15)]  # Example thresholds, adjust as needed
    for r_th, t_th in pose_recall_th:
        metrics[f'Recall ({r_th}deg, {t_th}cm)'] = []

    # Compute R/T errors
    err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)
    metrics['R error'].extend(err_R.tolist())
    metrics['T error'].extend(err_T.tolist())

    # Compute pose recall
    for r_th, t_th in pose_recall_th:
        succ_r, succ_t = err_R <= r_th, err_T <= t_th
        succ_pose = np.logical_and(succ_r, succ_t).astype(float)
        metrics[f'Recall ({r_th}deg, {t_th}cm)'].extend(succ_pose.tolist())

    # Process object symmetries
    obj_sym = format_sym_set(obj_symms[obj_f])
    add_diam = get_diameter(obj_model['pts']) / 1000.

    # Compute ADD(S)
    if obj_sym.shape[0] > 1:
        adds = compute_adds(obj_model['pts'] / 1000., pred_q, gt_pose_q)
    else:
        adds = compute_add(obj_model['pts'] / 1000., pred_q, gt_pose_q)
    metrics['ADD(S)-0.1d'].append(float(adds <= add_diam * 0.1))

    # Convert poses to float16 and separate R/t
    pred_q, gt_pose_q = pred_q.astype(np.float16), gt_pose_q.astype(np.float16)
    pred_r, pred_t = pred_q[:3, :3], np.expand_dims(pred_q[:3, 3], axis=1) * 1000
    gt_r, gt_t = gt_pose_q[:3, :3], np.expand_dims(gt_pose_q[:3, 3], axis=1) * 1000

    # Compute BOP metrics
    mspd_err = my_mspd(pred_r, pred_t, gt_r, gt_t, K.reshape(3, 3), obj_model['pts'], obj_sym)
    mssd_err = my_mssd(pred_r, pred_t, gt_r, gt_t, obj_model['pts'], obj_sym)

    # Define recall thresholds
    mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    # Compute MSSD and MSPD metrics
    mssd_cur_rec = mssd_rec * obj_diam
    mean_mssd = (mssd_err < mssd_cur_rec).mean()
    mean_mspd = (mspd_err < mspd_rec).mean()
    metrics['MSSD'].append(mean_mssd)
    metrics['MSPD'].append(mean_mspd)

    # Compute VSD metrics
    vsd_delta = 15.0
    vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    vsd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

    vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, depth_q, K.reshape(3, 3),
                   vsd_delta, vsd_taus, True, obj_diam, renderer, obj_f)
    vsd_errs = np.asarray(vsd_errs)
    all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
    mean_vsd = all_vsd_recs.mean()
    metrics['VSD'].append(mean_vsd)

    # Compute average recall (AR)
    metrics['AR'].append((mean_mssd + mean_mspd + mean_vsd) / 3.)

    # Store object and instance IDs
    metrics['cls_id'].append(obj_f)
    metrics['instance_id'].append(instance_id)

    return metrics


def compute_pose_metrics(metrics, pred_q, gt_pose_q, obj_model, obj_diam, obj_symms, depth_q, K, renderer, obj_f, instance_id):
    """
    Compute various pose estimation metrics including R/T errors, ADD(S), MSSD, MSPD, and VSD.

    Args:
        pred_q: Predicted pose matrix
        gt_pose_q: Ground truth pose matrix
        obj_model: Object model dictionary containing 'pts' key
        obj_diam: Object diameter
        obj_symms: Object symmetries
        depth_q: Depth image
        K: Camera intrinsic matrix
        renderer: Renderer object for VSD computation
        obj_f: Object ID/filename
        instance_id: Instance ID

    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Add pose recall metrics dynamically
    pose_recall_th = [(5, 5), (10, 10), (15, 15)]  # Example thresholds, adjust as needed
    for r_th, t_th in pose_recall_th:
        metrics[f'Recall ({r_th}deg, {t_th}cm)'] = []

    # Compute R/T errors
    err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)
    metrics['R error'].extend(err_R.tolist())
    metrics['T error'].extend(err_T.tolist())

    # Compute pose recall
    for r_th, t_th in pose_recall_th:
        succ_r, succ_t = err_R <= r_th, err_T <= t_th
        succ_pose = np.logical_and(succ_r, succ_t).astype(float)
        metrics[f'Recall ({r_th}deg, {t_th}cm)'].extend(succ_pose.tolist())

    # Process object symmetries
    obj_sym = format_sym_set(obj_symms[obj_f])
    add_diam = get_diameter(obj_model['pts']) / 1000.

    # Compute ADD(S)
    if obj_sym.shape[0] > 1:
        adds = compute_adds(obj_model['pts'] / 1000., pred_q, gt_pose_q)
    else:
        adds = compute_add(obj_model['pts'] / 1000., pred_q, gt_pose_q)
    metrics['ADD(S)-0.1d'].append(float(adds <= add_diam * 0.1))

    # Convert poses to float16 and separate R/t
    pred_q, gt_pose_q = pred_q.astype(np.float16), gt_pose_q.astype(np.float16)
    pred_r, pred_t = pred_q[:3, :3], np.expand_dims(pred_q[:3, 3], axis=1) * 1000
    gt_r, gt_t = gt_pose_q[:3, :3], np.expand_dims(gt_pose_q[:3, 3], axis=1) * 1000

    # Compute BOP metrics
    mspd_err = my_mspd(pred_r, pred_t, gt_r, gt_t, K.reshape(3, 3), obj_model['pts'], obj_sym)
    mssd_err = my_mssd(pred_r, pred_t, gt_r, gt_t, obj_model['pts'], obj_sym)

    # Define recall thresholds
    mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    # Compute MSSD and MSPD metrics
    mssd_cur_rec = mssd_rec * obj_diam
    mean_mssd = (mssd_err < mssd_cur_rec).mean()
    mean_mspd = (mspd_err < mspd_rec).mean()
    metrics['MSSD'].append(mean_mssd)
    metrics['MSPD'].append(mean_mspd)

    # Compute VSD metrics
    vsd_delta = 15.0
    vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    vsd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

    vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, depth_q, K.reshape(3, 3),
                   vsd_delta, vsd_taus, True, obj_diam, renderer, obj_f)
    vsd_errs = np.asarray(vsd_errs)
    all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
    mean_vsd = all_vsd_recs.mean()
    metrics['VSD'].append(mean_vsd)

    # Compute average recall (AR)
    metrics['AR'].append((mean_mssd + mean_mspd + mean_vsd) / 3.)

    # Store object and instance IDs
    metrics['cls_id'].append(obj_f)
    metrics['instance_id'].append(instance_id)

    return metrics

def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    elif dist_type == 'inv_norm_cosine':
        return 0.5 * (-1*cosine_similarity(A.unsqueeze(1), B.unsqueeze(0),dim=2) + 1)
    elif dist_type == 'cosine':
        return 0.5 * (cosine_similarity(A.unsqueeze(1), B.unsqueeze(0),dim=2) + 1)
    else:
        raise NotImplementedError('Not implemented')



def np_transform_pcd(pcd: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    pcd = pcd.astype(np.float16)
    r = r.astype(np.float16)
    t = t.astype(np.float16)
    rot_pcd = np.dot(np.asarray(pcd), r.T) + t
    return rot_pcd


def mask_iou(mask1: Tensor, mask2: Tensor) -> Tensor:
    '''
    Both input as [B,H,W]
    '''

    assert mask1.shape == mask2.shape
    assert len(mask1.shape) == 3

    B, H, W = mask1.shape

    mask1 = mask1.view(B, H * W)
    mask2 = mask2.view(B, H * W)

    union = torch.logical_or(mask1, mask2)
    inters = torch.logical_and(mask1, mask2)

    union_areas = union.sum(1)
    inter_areas = inters.sum(1)

    ious = inter_areas / union_areas

    return ious


def get_entropy(probs: torch.Tensor, dim: int, norm: bool = False) -> torch.Tensor:
    '''
    Compute entropy along a given dimension, optionally normalizing it between 0 and 1
    probs is supposed to be a distribution along the given dimension
    '''

    entropy = (-1 * torch.mul(probs, torch.log(probs + 1e-12))).sum(dim)

    if norm:
        size = probs.shape[dim]
        uniform_dist = Categorical(torch.ones(size) / size)
        max_entropy = uniform_dist.entropy()
        entropy = entropy / max_entropy

    return entropy


def compute_fmr(feats1: Tensor, feats2: Tensor, dist_th: float, inlier_th: float):
    '''
    Compute FMR between to correspondence sets (B,N,D), (B,N,D)
    '''
    assert feats1.shape == feats2.shape

    if len(feats1.shape) == 2:
        # handles unbatched case
        feats1, feats2 = feats1.unsqueeze(0), feats2.unsqueeze(0)

    # use inverted cosine similarity as distance metric
    dist_pos = .5 * (-1 * F.cosine_similarity(feats1, feats2, dim=2) + 1)
    # ratio of inlier (i.e. pairs under a distance in the feature space) of each corr set
    inlier_ratio = ((dist_pos < dist_th).to(float)).mean(1)

    # mean os success/failure for each corr set
    recall = (inlier_ratio > inlier_th).to(float)

    return recall


def pixel_match_loss(gt_matches: torch.Tensor, pred_matches: torch.Tensor) -> torch.Tensor:
    '''
    Return pixel match loss between two batches of predicted and ground truth matches (B,N,4)
    '''

    err = torch.zeros(pred_matches.shape[0])
    for i in range(pred_matches.shape[0]):
        gt_i, pred_i = gt_matches[i].cpu(), pred_matches[i]
        if gt_i.shape[0] > 0 and pred_i.shape[0] > 0:
            dists = (pdist(pred_i[:, :2], gt_i[:, :2]) + pdist(pred_i[:, 2:], gt_i[:, 2:])) / 2.
            r, c = linear_sum_assignment(dists)
            err[i] = dists[r, c].mean()
        else:
            err[i] = 0.

    return err


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    # determine the (x, y)-coordinates of the intersection rectangle

    # switch from (x,y,w,h) to (x1,y1,x2,y2)
    box1[2] = box1[2] + box1[0]
    box1[3] = box1[3] + box1[1]
    box2[2] = box2[2] + box2[0]
    box2[3] = box2[3] + box2[1]

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    boxBArea = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def np_mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    m1, m2 = m1.astype(bool), m2.astype(bool)
    union = np.logical_or(m1, m2).sum()
    inter = np.logical_and(m1, m2).sum()

    iou = float(inter) / float(union)

    return iou


def class_cosine_similarty(feats_dict: dict) -> np.ndarray:
    '''
    Given a dict of N classes, each a Tensor [P,D] with P number of elements and D dimension,
    compute similarity scores between classes using normalized cosine similarity.
    '''

    keys = list(feats_dict.keys())
    N_CLS = len(keys)
    mat_sim = torch.zeros((len(keys), len(keys)))
    for i1, obj1 in enumerate(keys):

        for i2, obj2 in enumerate(keys):

            if obj1 == obj2:
                cs = self_cosine_similarity(feats_dict[obj1])
                N_ELEM = feats_dict[obj1].shape[0]
                avg_sim = (cs.sum() - N_CLS) / (N_ELEM * (N_ELEM - 1))
            else:
                avg_sim = cross_cosine_similarity(feats_dict[obj1], feats_dict[obj2]).mean()
            mat_sim[i1, i2] = avg_sim

    # move cosine similariti to a [0,1] score
    mat_sim = (mat_sim + 1) / 2.

    return mat_sim.numpy()


def self_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    '''
    Given a [N,D] vector, returns [N,N] matrix with cosine similarity
    '''

    N = x.shape[0]

    c_matrix = torch.zeros((N, N))

    for i in range(N):
        c_matrix[i, :] = cosine_similarity(x, x[i, :])

    return c_matrix


def cross_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    '''
    Given two vectors [N1,D] and [N2,D], returns [N1,N2] cosine similarity matrix
    '''

    N1, N2 = x1.shape[0], x2.shape[0]
    assert x1.shape[1] == x2.shape[1]

    c_matrix = torch.zeros((N1, N2))
    for i in range(N2):
        c_matrix[i, :] = cosine_similarity(x1, x2[i, :])

    return c_matrix


def compute_add(pcd: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
    gt_r, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    # ADD computation
    add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

    return add


def compute_adds(pcd: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
    gt_r, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    # ADD-S computation
    kdt = KDTree(model_gt, metric='euclidean')
    distance, _ = kdt.query(model_pred, k=1)
    adds = np.mean(distance)

    return adds


def compute_RT_distances(pose1: np.ndarray, pose2: np.ndarray):
    '''
    :param RT_1: [B, 4, 4]. homogeneous affine transformation
    :param RT_2: [B, 4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    Works in batched or unbatched manner. NB: assumes that translations are in Meters
    '''

    if pose1 is None or pose2 is None:
        return -1

    if len(pose1.shape) == 2:
        pose1 = np.expand_dims(pose1, axis=0)
        pose2 = np.expand_dims(pose2, axis=0)

    try:
        assert np.array_equal(pose1[:, 3, :], pose2[:, 3, :])
        assert np.array_equal(pose1[0, 3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(pose1[:, 3, :], pose2[:, 3, :])

    BS = pose1.shape[0]

    R1 = pose1[:, :3, :3] / np.cbrt(np.linalg.det(pose1[:, :3, :3]))[:, None, None]
    T1 = pose1[:, :3, 3]

    R2 = pose2[:, :3, :3] / np.cbrt(np.linalg.det(pose2[:, :3, :3]))[:, None, None]
    T2 = pose2[:, :3, 3]

    R = np.matmul(R1, R2.transpose(0, 2, 1))
    arccos_arg = (np.trace(R, axis1=1, axis2=2) - 1) / 2
    arccos_arg = np.clip(arccos_arg, -1 + 1e-12, 1 - 1e-12)
    theta = np.arccos(arccos_arg) * 180 / np.pi
    theta[np.isnan(theta)] = 180.
    shift = np.linalg.norm(T1 - T2, axis=-1) * 100

    return theta, shift
