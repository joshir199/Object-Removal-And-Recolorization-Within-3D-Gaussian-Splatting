
import os
import cv2
import numpy as np
from collections import defaultdict
import argparse


def compute_iou_and_acc(gt_mask, pred_mask):
    """
    Compute IoU and Accuracy for given binary masks.
    """
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    correct = (gt_mask == pred_mask).sum()

    iou = intersection / union if union > 0 else 1.0
    acc = correct / gt_mask.size
    return iou, acc


def evaluate_dataset(dataset_root):
    """
    Compute meanIoU and meanAccuracy per scene and overall.
    """
    scene_results = defaultdict(lambda: {"ious": [], "accs": []})
    overall_ious, overall_accs = [], []

    for scene_name in os.listdir(dataset_root):
        if not scene_name.endswith("_gt"):
            continue

        scene_id = scene_name.replace("_gt", "")
        gt_scene_path = os.path.join(dataset_root, scene_name)
        pred_scene_path = os.path.join(dataset_root, scene_id + "_pred")

        for frame in os.listdir(gt_scene_path):
            gt_frame_path = os.path.join(gt_scene_path, frame)
            pred_frame_path = os.path.join(pred_scene_path, frame)

            if not os.path.exists(pred_frame_path):
                continue

            for mask_file in os.listdir(gt_frame_path):
                gt_mask_path = os.path.join(gt_frame_path, mask_file)
                pred_mask_path = os.path.join(pred_frame_path, mask_file)
                #print(f"gtr_mask_path: {gt_mask_path}")
                #print(f"pred_mask_path: {pred_mask_path} \n")

                if not os.path.exists(pred_mask_path):
                    continue

                # Load grayscale mask
                gt_mask_bin = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)  # gt image is grayscale
                pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE) # pred image is RGB

                _, pred_mask_bin = cv2.threshold(pred_mask, 240, 255, cv2.THRESH_BINARY_INV) # convert to binary

                iou, acc = compute_iou_and_acc(gt_mask_bin, pred_mask_bin)
                #print(f"iou: {iou}, acc: {acc} \n")

                scene_results[scene_id]["ious"].append(iou)
                scene_results[scene_id]["accs"].append(acc)
                overall_ious.append(iou)
                overall_accs.append(acc)

    # Collect results together
    results = {}
    for scene_id, vals in scene_results.items():
        results[scene_id] = {
            "meanIoU": np.mean(vals["ious"]) if vals["ious"] else 0,
            "meanAccuracy": np.mean(vals["accs"]) if vals["accs"] else 0,
        }

    results["overall"] = {
        "meanIoU": np.mean(overall_ious) if overall_ious else 0,
        "meanAccuracy": np.mean(overall_accs) if overall_accs else 0,
    }

    return results



if __name__ == "__main__":
    # create dataset in below format  
    """
    test_folder/
         sceneA_gt/
              frame_0001/
                  object_x.png
                  object_y.png
              frame_0002/
                  object_x.png
                  object_y.png
         sceneA_pred/
              frame_0001/
                  object_x.png
                  object_y.png
              frame_0002/
                  object_x.png
                  object_y.png
         sceneB_gt/
              frame_0001/
                  object_x.png
                  object_y.png
                  .
                  .
                  .
    """
    
    # run command: #python3 overall_metric_evaluation.py --test_folder /vol/teaching/DigitalHuman/Raushan/output/flashsplat_feature/test_dataset_results/test_dataset
    
    parser = argparse.ArgumentParser(description="evaluate the predicted mask.")
    parser.add_argument('--test_folder', type=str, required=True,
                        help='Path to the test folder containing gt and pred masks for different scenes.')
                        
    args = parser.parse_args()
    test_folder = args.test_folder
    
    results = evaluate_dataset(test_folder)
    for scene, metrics in results.items():
        print(f"{scene}: IoU={metrics['meanIoU']:.4f}, Acc={metrics['meanAccuracy']:.4f}")