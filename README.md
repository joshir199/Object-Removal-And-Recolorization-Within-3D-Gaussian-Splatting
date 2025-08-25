# Object Removal And Recolorization Within 3D Gaussian Splatting

## Introduction

This project introduces a novel framework for multi-view consistent 3D segmentation using 3D Gaussian Splatting and high-quality 2D segmentation features from SAM-HQ. Our approach enables real-time object removal and recolorization in dynamic 3D scenes, addressing challenges in mask quality and computational efficiency. By leveraging state-of-the-art 2D segmentation models and explicit 3D representations, we achieve high-precision segmentation with applications in virtual reality, robotics, and scene editing. The source code, dataset, and detailed results are available in this repository.

---

## Architecture

<figure>
  <img src="assets/object_feature_architecture.jpg" alt="Architecture Diagram 1" width="600">
  <figcaption>Figure 1: Overview of our object feature based 3D segmentation pipeline using 3D Gaussian Splatting.</figcaption>
</figure>

<figure>
  <img src="assets/multi_label_assignment.jpg" alt="Architecture Diagram 2" width="600">
  <figcaption>Figure 2: Detailed view of the feature distillation and mask-lifting modules.</figcaption>
</figure>

---

## Results

### Tabular Results

<figure>
  <img src="assets/SAM_PSNR_Comparison.jpg" alt="Table 1" width="600">
</figure>

<figure>
  <img src="assets/new_dataset_result.jpg" alt="Table 2" width="600">
</figure>

<figure>
  <img src="assets/NVOS_dataset_result_comparison.jpg" alt="Table 3" width="600">
</figure>

<figure>
  <img src="assets/object_influence.jpg" alt="Table 4" width="600">
</figure>

---

## Image Results

<figure>
  <img src="assets/screenshot_mask.jpg" alt="Result Image 3" width="600">
  <figcaption>Figure 3: Segmentation result showing precise object boundaries.</figcaption>
</figure>

<figure>
  <img src="assets/object_removal.jpg" alt="Result Image 4" width="600">
  <figcaption>Figure 4: Object removal using our method.</figcaption>
</figure>

<figure>
  <img src="assets/object_ectraction.jpg" alt="Result Image 5" width="600">
  <figcaption>Figure 5: Extraction of objects while maintaining multi-view consistency.</figcaption>
</figure>

<figure>
  <img src="assets/recolor_part1.jpg" alt="Result Image 6" width="600">
  <figcaption>Figure 6: Recolorization of objects  while maintaining multi-view consistency.</figcaption>
</figure>

<figure>
  <img src="assets/recolor_part2.jpg" alt="Result Image 7" width="600">
  <figcaption>Figure 7: Recolorization of objects  while maintaining multi-view consistency.</figcaption>
</figure>

---

[Watch demo video](assets/videos/bear_extracted.mp4)

---
