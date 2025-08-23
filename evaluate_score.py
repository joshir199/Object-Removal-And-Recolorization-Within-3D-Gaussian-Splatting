#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from os import path as osp
from tqdm import tqdm
from os import makedirs
# import imageio 
import cv2 
from gaussian_renderer import render
from gaussian_renderer import flashsplat_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from PIL import Image
import numpy as np
import colorsys

import pdb 

import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p
from scipy.spatial import cKDTree

def mean_neighborhood(input_img, N):

    pad = (N - 1) // 2
    padded_img = F.pad(input_img, (pad, pad, pad, pad), mode='constant', value=0)
    patches = padded_img.unfold(1, N, 1).unfold(2, N, 1)
    mean_patches = patches.mean(dim=-1).mean(dim=-1)
    return mean_patches


def save_mp4(dir, fps):
    imgpath = dir
    frames = []
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
    fps = float(fps)
    for name in sorted(os.listdir(imgpath)):
        img = osp.join(imgpath, name)
        img = cv2.imread(img)
        frames.append(img)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(os.path.join(dir,'eval.mp4'), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

def get_object_mean_xyz(xyz_cpu, object_mask):
    """
    Compute the mean (x, y, z) coordinate of Gaussians for a given object mask.
    
    """
    object_mask = object_mask.bool().to("cpu") # [N]
    selected_points = xyz_cpu[object_mask]  # [M, 3]
    
    if selected_points.size == 0:
        raise ValueError("No points found for the given mask.")

    mean_xyz = torch.tensor(selected_points).mean(dim=0)  # [3]
    print("mean_xyz value: ", mean_xyz)
    return mean_xyz

def find_nearest_gaussians_kdtree(xyz_cpu, query_point, k=20):

    tree = cKDTree(xyz_cpu)
    distances, indices = tree.query(query_point, k=k)
    return indices


def find_objectId_from_3Dpoints(gaussians_xyz_cpu, object_pred, query_point):
    """
    Find the object Id using k-nearest neighbours (100).

    Args:
        gaussians: Gaussian object with _features_dc storing colors in SH representation.
        object_pred (Tensor): prediction_objects probability of shape [256, num_points] 
        indicating which Gaussians belongs to which object Id.
        query_point (tuple/torch.Tensor): Target 3D points  in [x, y, z] coordinate
    """
    object_pred = object_pred.to("cpu") # [256, N]
    initial_labels = torch.argmax(object_pred, dim=0) # [N]
    indices = find_nearest_gaussians_kdtree(gaussians_xyz_cpu, query_point)
    
    object_ids = initial_labels[indices]
    unique_ids, counts = torch.unique(object_ids, return_counts=True)
    max_idx = torch.argmax(counts)

    mode_id = unique_ids[max_idx].item()
    print("predicted object Id: ", mode_id)
    return mode_id
    

def recolor_gaussians_by_mask(gaussians, mask, new_rgb=(1.0, 0.0, 0.0)):
    """
    Change the base (DC) color of Gaussians that match mask.

    Args:
        gaussians: Gaussian object with _features_dc storing colors in SH representation.
        mask (Tensor): Boolean mask of shape [num_points] indicating which Gaussians to recolor.
        new_rgb (tuple/list/torch.Tensor): Target RGB color in [0, 1] range, shape [3]. Default Red
    """
    mask = mask.bool().to(gaussians._features_dc.device)
    
    # Convert new color to tensor
    new_rgb = torch.tensor(new_rgb, dtype=gaussians._features_dc.dtype, device=gaussians._features_dc.device)

    # _features_dc: [num_points, 1, 3] ? only first SH coefficient stores base color
    features_dc = gaussians._features_dc.clone()
    features_dc[mask, 0, :] = new_rgb  # Update only masked points' base RGB into new color
    gaussians._features_dc = torch.nn.Parameter(features_dc)
    

def filter_gaussians_by_mask(gaussians, remain_mask):
    # Ensure mask is on CPU and boolean
    remain_mask = remain_mask.bool().cpu()

    # Apply filtering to all relevant parameters
    gaussians._xyz = torch.nn.Parameter(gaussians._xyz[remain_mask])
    gaussians._opacity = torch.nn.Parameter(gaussians._opacity[remain_mask])
    gaussians._scaling = torch.nn.Parameter(gaussians._scaling[remain_mask])
    gaussians._rotation = torch.nn.Parameter(gaussians._rotation[remain_mask])
    gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc[remain_mask])
    gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[remain_mask])

    # For object features
    gaussians._object_feature = torch.nn.Parameter(gaussians._object_feature[remain_mask])
    

def construct_list_of_attributes(gaussians):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(gaussians._features_dc.shape[1] * gaussians._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(gaussians._features_rest.shape[1] * gaussians._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(gaussians._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gaussians._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    # Add object features
    for i in range(gaussians._object_feature.shape[1] * gaussians._object_feature.shape[2]):
        l.append('obj_feat_{}'.format(i))
    return l
        
def save_edited_ply(gaussians, path, task="remove"):
    path = os.path.join(path, '{}_point_cloud.ply'.format(task))
    mkdir_p(os.path.dirname(path))

    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()
    object_feature = gaussians._object_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    attribute_list = construct_list_of_attributes(gaussians)
    print("attributes to be saved: ", attribute_list)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gaussians)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, object_feature), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print("Edited Gaussians ply file saved: ")
    

def multi_instance_opt_prior_cpu_new(all_counts, object_pred, slackness=0.0, task="recolor", gamma=1.0, conf_threshold=0.25, penalty_k=3.0):
    # Ensure tensor is on CPU
    all_counts = all_counts.to("cpu")
    object_pred = object_pred.to("cpu")

    # Normalize counts across classes for each point
    all_counts = torch.nn.functional.normalize(all_counts, dim=0)
    all_counts_sum = all_counts.sum(dim=0) # [N]

    # Predicted labels and scores for each Gaussian
    initial_labels = torch.argmax(object_pred, dim=0).to("cpu")   # [N]
    pred_scores = torch.max(object_pred, dim=0)[0].to("cpu")      # [N]

    all_obj_labels = torch.zeros_like(all_counts).to("cpu")       # [num_classes, N]
    obj_num = all_counts.size(0)

    for obj_idx, obj_counts in enumerate(tqdm(all_counts, desc="multi-view optimize")):
        if obj_counts.sum().item() == 0:
            continue

        # Stack: [not-object score, object score]
        obj_counts = torch.stack([all_counts_sum - obj_counts, obj_counts], dim=0)
        obj_counts = torch.nn.functional.normalize(obj_counts, dim=0)

        # Apply slackness bias to all "not object" scores
        obj_counts[0, :] += -0.4

        # Prior term (extra score adjustments)
        prior_term = torch.zeros_like(obj_counts).to("cpu")  # Shape [2, N]

        # Positive prior: boost score for points predicted as this object
        valid_prior = (initial_labels == obj_idx)
        prior_term[1, valid_prior] = gamma * pred_scores[valid_prior]

        # Penalty for low-confidence predictions for this object during object removal
        if task == "object_removal":
            penalty_mask = (object_pred[obj_idx, :] < conf_threshold)  # score for obj_idx
            prior_term[0, penalty_mask] = -penalty_k * conf_threshold

        # Add priors
        obj_counts = obj_counts + prior_term

        # Assign labels for this object index
        obj_label = obj_counts.max(dim=0)[1]
        all_obj_labels[obj_idx] = obj_label

    return all_obj_labels


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier,
               slackness=0, view_num=-1, obj_num=1, obj_id=0, task="object_removal"):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "eval_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "eval_gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    view_num = len(views)
    views_used = views

    all_counts = None
    stats_counts_path = os.path.join(model_path, name, "eval_ours_{}".format(iteration), "used_count")
    makedirs(stats_counts_path, exist_ok=True)
    cur_count_dir = os.path.join(stats_counts_path, "eval_view_num_{:03d}_objnum_{:03d}_ids_{:03d}.pth".format(view_num, obj_num, obj_id))
    print("cur_count_dir: ", cur_count_dir)
    

    print("render using flashsplat_render ")
    for idx, view in enumerate(tqdm(views_used, desc="Rendering progress")):
        if obj_num == 1:
            # making multi-label mask to binary mask for object_Id k
            gt_mask_ori = view.objects
            gt_mask = torch.where(gt_mask_ori == obj_id, torch.tensor(255, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))
            gt_mask = gt_mask.to(torch.float32) / 255.
        else:
            gt_mask = view.objects.to(torch.float32)
            assert torch.any(torch.max(gt_mask) <= obj_num), f"max_obj {int(torch.max(gt_mask).item())}"
        
        render_pkg = flashsplat_render(view, gaussians, pipeline, background, gt_mask=gt_mask, obj_num=obj_num)
        rendering = render_pkg["render"]
        used_count = render_pkg["used_count"]
        
        if all_counts is None:
            all_counts = torch.zeros_like(used_count)
        gt = view.original_image[0:3, :, :]
        #torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        all_counts += used_count
        
    #save_mp4(render_path, 3)
    torch.save(all_counts, cur_count_dir)

    if all_counts is not None:

        # gaussians._object_feature.shape # [N, 1, 16]
        pred_object = classifier(gaussians._object_feature.permute(2, 0, 1).unsqueeze(0)).squeeze(-1) # input [batch, 16, N, 1]
        pred_object_scores = F.softmax(pred_object, dim=1)  # probability score
        print("pred_object shape: ", pred_object.shape) # [batch, 256, N]
        
        if obj_num == 1:
            # for binary seg,
            object_pred = pred_object_scores.squeeze(0).to("cpu")  #[256, N]
            initial_labels = torch.argmax(object_pred, dim=0).to("cpu")   # [N]
            pred_scores = torch.max(object_pred, dim=0)[0].to("cpu")      # [N]
            prior_term = torch.zeros_like(all_counts).to("cpu")  # Shape [2, N]
            
            valid_prior = (initial_labels == obj_id)
            prior_term[1, valid_prior] += 0.1 * pred_scores[valid_prior]  # gamma = 1.0
            penalty_mask = (object_pred[obj_id, :] < 0.50)  # conf_score = 0.25
            #prior_term[0, penalty_mask] += -1.0 * 0.25
            
            all_counts = torch.nn.functional.normalize(all_counts, dim=0)  # [2, N]
            all_counts[0, :] += slackness # slackness
            all_counts = all_counts + prior_term.cuda()  # prior term
            unique_label = all_counts.max(dim=0)[1]
        else:
            all_obj_labels = multi_instance_opt_prior_cpu_new(all_counts, pred_object.squeeze(0), slackness, task).cuda()
        

        render_num = all_counts.size(0)  # object Ids to be evaluated
        views_used = views
        
        for obj_idx in range(render_num):
            #if obj_id > 0:
            #    obj_idx = obj_id 
            #    early_stop = True
            if obj_num == 1:
                obj_used_mask = (unique_label == obj_idx)
            else:
                obj_used_mask = (all_obj_labels[obj_idx]).bool()
                if obj_used_mask.sum().item() == 0:
                    continue
            
            obj_render_path = os.path.join(model_path, name, "eval_ours_{}".format(iteration), "obj_{:03d}_{:01d}".format(obj_id, obj_idx))
            os.makedirs(obj_render_path, exist_ok=True)

            for idx, view in enumerate(tqdm(views_used, desc="Rendering eval object {:03d}".format(obj_idx))):
                render_pkg = flashsplat_render(view, gaussians, pipeline, background, used_mask=obj_used_mask)
                rendering = render_pkg["render"]
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(rendering, os.path.join(obj_render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                
            save_mp4(obj_render_path, 3)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, 
                slackness : float, view_num : int, obj_num : int, obj_id: int, task: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        num_classes = 256
        print("Num classes: ",num_classes)
        classifier = torch.nn.Conv2d(gaussians.num_object_feature, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, 
                        slackness, view_num, obj_num, obj_id, task)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, 
                        slackness, view_num, obj_num, obj_id, task)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--slackness", default=0.0, type=float)
    parser.add_argument("--view_num", default=10.0, type=int)
    parser.add_argument("--obj_num", default=1, type=int)
    parser.add_argument("--obj_id", default=-1, type=int)
    parser.add_argument("--task", default="object_removal", type=str, help="Can be object_removal or recolor task")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # args.object_path = args.object_mask

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.slackness, args.view_num, args.obj_num, args.obj_id, args.task)