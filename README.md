# Object Removal And Recolorization Within 3D Gaussian Splatting

## Introduction

This project introduces a novel framework for multi-view consistent 3D segmentation using 3D Gaussian Splatting and high-quality 2D segmentation features from SAM-HQ. Our approach enables real-time object removal and recolorization in dynamic 3D scenes, addressing challenges in mask quality and computational efficiency. By leveraging state-of-the-art 2D segmentation models and explicit 3D representations, we achieve high-precision segmentation with applications in virtual reality, robotics, and scene editing. The source code, dataset, and detailed results are available in this repository.

---
## Dataset

The 3D Segmentation HQ dataset is a curated collection of 5 real-world scenes with high-quality object segmentation masks designed for research in 3D scene understanding, editing, and rendering. This dataset improves upon existing benchmarks by providing cleaner and more consistent object masks across multiple views, enabling reliable evaluation and training for tasks such as:

- 3D semantic segmentation
- Object-level scene editing (e.g., removal, recolorization)
- 3D Gaussian Splatting with semantic supervision

Please find the dataset here: ![huggingface link](https://huggingface.co/datasets/joshir/3D-Scene-Segmentation-HQ)

---
## Architecture

<figure>
  <img src="assets/object_feature_architecture.jpg" alt="Architecture Diagram 1" width="600">
  <figcaption>Figure 1: Overview of our object feature based 3D segmentation pipeline using 3D Gaussian Splatting.</figcaption>
</figure>

&nbsp;

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

&nbsp;

<figure>
  <img src="assets/new_dataset_result.jpg" alt="Table 2" width="600">
</figure>

&nbsp;

<figure>
  <img src="assets/NVOS_dataset_result_comparison.jpg" alt="Table 3" width="600">
</figure>

&nbsp;

<figure>
  <img src="assets/object_influence.jpg" alt="Table 4" width="600">
</figure>

---

## Image Results

<figure>
  <img src="assets/screenshot_mask.jpg" alt="Result Image 3" width="600">
  <figcaption>Figure 3: Segmentation result showing precise object boundaries.</figcaption>
</figure>

&nbsp;

  
<figure>
  <img src="assets/object_removal.jpg" alt="Result Image 4" width="600">
  <figcaption>Figure 4: Object removal using our method.</figcaption>
</figure>

&nbsp;
&nbsp;
  
<figure>
  <img src="assets/object_ectraction.jpg" alt="Result Image 5" width="600">
  <figcaption>Figure 5: Extraction of objects while maintaining multi-view consistency.</figcaption>
</figure>

  &nbsp;
  
<figure>
  <img src="assets/recolor_part1.jpg" alt="Result Image 6" width="600">
  <figcaption>Figure 6: Recolorization of objects  while maintaining multi-view consistency.</figcaption>
</figure>

&nbsp;

  
<figure>
  <img src="assets/recolor_part2.jpg" alt="Result Image 7" width="600">
  <figcaption>Figure 7: Recolorization of objects  while maintaining multi-view consistency.</figcaption>
</figure>

---
# Video results

Following results shows the ooutput of editing task like recolor, object removal and object extraction on 3D scene. 
The displayed results covers the quality by rendering all the views for various scene and tasks mentioned as "scene_name [task_name]"


<table>
  <tr>
    <td align="center">
      <img src="assets/gifs/bear_recolor.gif" width="280"><br>
      <b>bear [recolor]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/bear_removed.gif" width="280"><br>
      <b>bear [removed]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/bear_extracted.gif" width="280"><br>
      <b>bear [extracted]</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/gifs/fortress_recolor.gif" width="280"><br>
      <b>fortress [recolor]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/fortress_removed.gif" width="280"><br>
      <b>fortress [removed]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/fortress_extracted.gif" width="280"><br>
      <b>fortress [extracted]</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/gifs/horn_recolor.gif" width="280"><br>
      <b>horn [recolor]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/horn_removed.gif" width="280"><br>
      <b>horn [removed]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/horn_extracted.gif" width="280"><br>
      <b>horn [extracted]</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/gifs/ramen_recolor.gif" width="280"><br>
      <b>ramen [recolor]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/ramen_removed.gif" width="280"><br>
      <b>ramen [removed]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/ramen_extracted.gif" width="280"><br>
      <b>ramen [extracted]</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/gifs/teatime_recolor.gif" width="280"><br>
      <b>teatime [recolor]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/teatime_removed.gif" width="280"><br>
      <b>teatime [removed]</b>
    </td>
    <td align="center">
      <img src="assets/gifs/teatime_extracted.gif" width="280"><br>
      <b>teatime [extracted]</b>
    </td>
  </tr>
</table>


---

## Reference Links

This project is built upon the great works of previous projects in the field of 3D Gaussian Splatting research.
I want to acknowledge and thanks them for keeping it open-source for future research works.

1. Original Gaussian splatting : https://github.com/graphdeco-inria/gaussian-splatting
2. Track anything with DEVA : https://github.com/hkchengrex/Tracking-Anything-with-DEVA
3. FlashSplat : https://github.com/florinshen/FlashSplat
4. Segment Anything in High Quality : https://github.com/SysCV/sam-hq

---
