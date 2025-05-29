<div align="center">

<h1>PanopticNeRF-360: Panoramic 3D-to-2D Label Transfer in Urban Scenes</h1>

<div>
    <a href='https://fuxiao0719.github.io/' target='_blank'>Xiao Fu</a><sup>1</sup>&emsp;
    Shangzhan Zhang<sup>1</sup>&emsp;
    <a href="https://tianrun-chen.github.io/" target="_blank">Tianrun Chen</a><sup>1</sup>&emsp;
    Yichong Lu<sup>1</sup>&emsp;
    <a href='https://xzhou.me/' target='_blank'>Xiaowei Zhou</a><sup>1</sup>&emsp;
    <a href='http://www.cvlibs.net/' target='_blank'>Andreas Geiger</a><sup>2</sup>&emsp;
    <a href='https://yiyiliao.github.io/' target='_blank'>Yiyi Liao</a><sup>1†</sup>
</div>
<div>
    <sup>1</sup>Zhejiang University&emsp;
    <sup>2</sup>University of Tübingen and Tübingen AI Center
</div>
<div>
    <sup>+</sup>corresponding author
</div>

<strong>arXiv 2023 </strong>

<h4 align="center">
  <a href="https://arxiv.org/pdf/2309.10815.pdf" target='_blank'>[Paper]</a> •
  <a href="https://fuxiao0719.github.io/projects/panopticnerf360/" target='_blank'>[Project Page]</a> •
  <a href="http://www.cvlibs.net/datasets/kitti-360/" target='_blank'>[Dataset]</a>
</h4>

</div>

## Installation
1. Create a virtual environment via `conda`. 
    ```
    conda env create -f environment.yml
    conda activate panopticnerf360
    ```
2. Install tiny-cuda-nn environment (C++/CUDA APIs) using [official guide](https://github.com/NVlabs/tiny-cuda-nn).

## Data Preparation
1. We evaluate our model on [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/). Here we show the structure of dataset as follow. You can download it from [here](https://drive.google.com/file/d/1A4bGVXmdRbubQNHX75rK1w26OnWIfNkM/view?usp=sharing) and then put it into `$ROOT`. In the `datasets`, we additionally provide some evaluation files on different scenes.
    ```
    ├── KITTI-360
      ├── 2013_05_28_drive_0000_sync
      ├── bbx_intersection
      ├── calibration
      ├── data_3d_bboxes
      ├── data_poses
      ├── fisheye
      ├── gt_2d_panoptics
      ├── gt_2d_semantics
      ├── lidar_depth
      ├── pspnet
      ├── tao_fisheye
      ├── sgm
      ├── visible_id
    ```

    | file | Intro |
    | ------ | ------ |
    | `2013_05_28_drive_0000_sync` | stereo perspective/two-side fisheye RGB images |
    | `bbx_intersection` | ray-mesh intersections, containing depths between hitting points and camera origin, semantic label IDs and bounding primitive IDs|
    | `calibration` | extrinsics and intrinsics of cameras |
    | `data_3d_bboxes` | bounding box primitives |
    | `data_poses` | system poses in a global Euclidean coordinate |
    | `fisheye` | fisheye related files |
    | `pspnet` | 2D pseudo semantic ground truth on perpsective views |
    | `tao_fisheye` | 2D pseudo semantic ground truth on fisheye views |
    | `sgm` | weak stereo depth supervision |
    | `visible_id` | per-frame visible bounding primitive IDs |

2. Generate ray-mesh intersections (`bbx_intersection/*.npz`). For the given test scene, `START=1908`, `NUM=64`.
    ```
    # image_00 (perspective)
    python mesh_intersection.py intersection_start_frame ${START} intersection_frames ${NUM} use_stereo False
    # image_01 (perspective)
    python mesh_intersection.py intersection_start_frame ${START} intersection_frames ${NUM} use_stereo True
    # image_02 (fisheye)
    python mesh_intersection_fisheye.py intersection_start_frame ${START} intersection_frames ${NUM} use_stereo False
    # image_03 (fisheye)
    python mesh_intersection_fisheye.py intersection_start_frame ${START} intersection_frames ${NUM} use_stereo True
    ```

## Training and Visualization
1. We provide the training code. Replace `resume False` with `resume True` to load the pretained model.
    ```
    python train_net.py --cfg_file configs/panopticnerf360_test.yaml use_stereo True gpus "0," resume False
    ```

2. Render semantic map, panoptic map, depth map, and normal map in a single forward pass. Please make sure to maximize the GPU memory utilization by increasing the size of the chunk to reduce inference time.
    ```
    python run.py --type visualize --cfg_file configs/panopticnerf360_test.yaml use_stereo True use_post_processing False gpus "0," 
    ```
3. When performing instance finetuning in a scene (here we only perform it on one scene, `scene`=`panopticnerf360_seq0_6398_6461_init`), run script
    ```
    sh instance_finetuning.sh
    ```
   Then render the corresponding output
    ```
    python run.py --type visualize --cfg_file configs/${scene}.yaml exp_name "${scene}_ft" use_stereo True use_post_processing False gpus "0," 
    python run.py --type visualize --cfg_file configs/${scene}.yaml use_stereo True use_post_processing False gpus "0," merge_instance True
    ```
4. Visualize novel view appearance & label synthesis on 360&deg; outward rotated views. (take frame `1930` for example)
    ```
    sh rotated_trajectory.sh
    ```
5. Visualize novel view appearance & label synthesis on panoramic view. (take frame `1947` for example)
    ```
    sh panorama.sh
    ```

## Evaluation
  ```
  ├── KITTI-360
    ├── gt_2d_semantics
    ├── gt_2d_panoptics
    ├── lidar_depth
  ```
1. Download the released [pretrained model](https://drive.google.com/drive/folders/19CVMmp_LkAs_wXPZkNNwKwwZNAdVFVRX?usp=sharing) and put it to `$ROOT/data/trained_model/panopticnerf360/panopticnerf360_test/latest.pth`.

2. We provide some semantic & panoptic GTs and LiDAR point clouds for evaluation. The details of evaluation metrics can be found in the paper.
3. Eval mean intersection-over-union (mIoU).
  ```
  # perspective
  python run.py --type eval_miou --cfg_file configs/panopticnerf360_test.yaml
  # fisheye
  python run.py --type eval_miou_fe --cfg_file configs/panopticnerf360_test.yaml
  ```

4. Eval panoptic quality (PQ)
  ```
  python run.py --type process_json --cfg_file configs/panopticnerf360_test.yaml
  python run.py --type process_json_gt --cfg_file configs/panopticnerf360_test.yaml
  # perspective
  python run.py --type eval_pq --cfg_file configs/panopticnerf360_test.yaml
  # fisheye
  python run.py --type eval_pq_fe --cfg_file configs/panopticnerf360_test.yaml
  ```
5. Eval depth with 0-100m LiDAR point clouds, where the far depth can be adjusted to evaluate the closer scene.
  ```
  python run.py --type eval_depth --cfg_file configs/panopticnerf360_test.yaml
  ```
6. Eval Peak Signal-to-Noise Ratio (PSNR)
  ```
  python run.py --type eval_psnr --cfg_file configs/panopticnerf360_test.yaml
  ```
7. Eval Multi-view Consistency (MC)
  ```
  python eval_consistency.py --cfg_file configs/panopticnerf360_test.yaml consistency_thres 0.1
  ```

## Citation

```bibtex
@article{fu2023panoptic,
  title={PanopticNeRF-360: Panoramic 3D-to-2D Label Transfer in Urban Scenes},
  author={Fu, Xiao and Zhang, Shangzhan and Chen, Tianrun and Lu, Yichong and Zhu, Lanyun and Zhou, Xiaowei and Geiger, Andreas and Liao, Yiyi},
  journal = {arxiv},
  year = {2023}
}
```
Copyright © 2023, Zhejiang University. All rights reserved. We favor any positive inquiry, please contact `lemonaddie0909@gmail.com`.
