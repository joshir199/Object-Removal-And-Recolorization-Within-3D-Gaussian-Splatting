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
    

def label_reassignment_with_prior_cpu(all_counts, object_pred, slackness=0.0, task="recolor", gamma=0.1, conf_threshold=0.25, penalty_k=3.0):
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
        obj_counts[0, :] += slackness

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


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb
    

# for visualize color mask
def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier,
               slackness=0, view_num=-1, obj_num=255, obj_ids=[], task="object_removal"):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if view_num > 0:
        view_idx = np.linspace(0, len(views) - 1, view_num, dtype=int).tolist()
        views_used = [views[idx] for idx in view_idx]
    else:
        view_num = len(views)
        views_used = views

    all_counts = None
    stats_counts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "used_count")
    makedirs(stats_counts_path, exist_ok=True)
    cur_count_dir = os.path.join(stats_counts_path, "eval_view_num_{:03d}_objnum_{:03d}.pth".format(view_num, obj_num))
    print("cur_count_dir: ", cur_count_dir)
    
    if os.path.exists(cur_count_dir):
        all_counts = torch.load(cur_count_dir).cuda()
    else:
        print("render using flashsplat_render ")
        for idx, view in enumerate(tqdm(views_used, desc="Rendering progress")):
            if obj_num == 1:
                gt_mask = view.objects.to(torch.float32) / 255.
            else:
                gt_mask = view.objects.to(torch.float32)
                #print("gt_mask_shape: ", gt_mask.shape)
                assert torch.any(torch.max(gt_mask) <= obj_num), f"max_obj {int(torch.max(gt_mask).item())}"
            
            render_pkg = flashsplat_render(view, gaussians, pipeline, background, gt_mask=gt_mask, obj_num=obj_num)
            rendering = render_pkg["render"]
            used_count = render_pkg["used_count"]
            
            if all_counts is None:
                all_counts = torch.zeros_like(used_count)
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
            all_counts += used_count
        save_mp4(render_path, 3)
        torch.save(all_counts, cur_count_dir)

    if all_counts is not None:
      
        # gaussians._object_feature.shape # [N, 1, 16]
        pred_object = classifier(gaussians._object_feature.permute(2, 0, 1).unsqueeze(0)).squeeze(-1) # input [batch, 16, N, 1]
        pred_object_scores = F.softmax(pred_object, dim=1)  # probability score
        print("pred_object shape: ", pred_object.shape) # [batch, 256, N]
        
        
        mean_xyz = np.array([0.0128, 0.6468, 2.1148])
        obj_idd = find_objectId_from_3Dpoints(gaussians._xyz.detach().cpu().numpy(), pred_object_scores.squeeze(0), mean_xyz)
        #print("all_obj_labels: ", all_obj_labels.shapelen)
        all_obj_labels = label_reassignment_with_prior_cpu(all_counts, pred_object_scores.squeeze(0), slackness, task).cuda()
        #print("all_obj_labels: ", all_obj_labels.shape)
        
        del all_counts, pred_object, pred_object_scores

        views_used = views

        # find 3d gaussians id to be removed 
        assert isinstance(obj_ids, list), "the type of obj_id shall be list."
        
        if task == "object_removal":
            print("objectIds: ", obj_ids)
            removal_mask = torch.zeros(all_obj_labels.shape[1], dtype=torch.bool, device=all_obj_labels.device)
    
            removal_name = ""
            for obj_idx in obj_ids:
                removal_mask[all_obj_labels[obj_idx].bool()] = True 
                removal_name += "_{:03d}".format(obj_idx)
            remain_mask = ~removal_mask 
            print("removal_mask shape: ", removal_mask.shape)
            xyz = get_object_mean_xyz(gaussians._xyz.detach().cpu().numpy(), removal_mask.cpu())
            #print("remain_mask shape: ", remain_mask.shape.len)
            
    
            obj_edit_path = os.path.join(model_path, name, "ours_{}".format(iteration), "new_remove_{}".format(removal_name))
            os.makedirs(obj_edit_path, exist_ok=True)
    
            for idx, view in enumerate(tqdm(views_used, desc="Rendering removal")):
                render_pkg = flashsplat_render(view, gaussians, pipeline, background, used_mask=remain_mask)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering, os.path.join(obj_edit_path, '{0:05d}'.format(idx) + ".png"))
                
            save_mp4(obj_edit_path, 3)
            
            print("editing gaussians: ")
            filter_gaussians_by_mask(gaussians, remain_mask)
        
        elif task == "recolor":
            removal_mask = torch.zeros(all_obj_labels.shape[1], dtype=torch.bool, device=all_obj_labels.device)
    
            removal_name = ""
            for obj_idx in obj_ids:
                removal_mask[all_obj_labels[obj_idx].bool()] = True 
                removal_name += "_{:03d}".format(obj_idx)
            remain_mask = ~removal_mask 
            print("removal_mask shape: ", removal_mask.shape)
    
            obj_edit_path = os.path.join(model_path, name, "ours_{}".format(iteration), "new_recolor_{}".format(removal_name))
            os.makedirs(obj_edit_path, exist_ok=True)
            
            recolor_gaussians_by_mask(gaussians, removal_mask, (0.0, 1.0, 0.0))
            del remain_mask, removal_mask
            color_mask = torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device=all_obj_labels.device)
    
            for idx, view in enumerate(tqdm(views_used, desc="Rendering recolor")):
                render_pkg = flashsplat_render(view, gaussians, pipeline, background, used_mask=color_mask)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering, os.path.join(obj_edit_path, '{0:05d}'.format(idx) + ".png"))
               
            save_mp4(obj_edit_path, 3)
            
        print("number of remained gaussians: ", gaussians._xyz.shape[0])
        save_edited_ply(gaussians, obj_edit_path, task)



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
    parser.add_argument('--obj_id', nargs='+', type=int, help='A list of integers')
    parser.add_argument("--task", default="object_removal", type=str, help="Can be object_removal or recolor task")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # args.object_path = args.object_mask

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.slackness, args.view_num, args.obj_num, args.obj_id, args.task)