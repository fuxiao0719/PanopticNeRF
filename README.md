# Panoptic NeRF
### [Project Page](https://fuxiao0719.github.io/projects/panopticnerf/) | [Paper](http://arxiv.org/abs/2203.15224) | [Dataset](http://www.cvlibs.net/datasets/kitti-360/)
<br/>      

> [Panoptic NeRF: 3D-to-2D Label Transfer for Panoptic Urban Scene Segmentation](http://arxiv.org/abs/2203.15224)  
> Xiao Fu*, Shangzhan zhang*, Tianrun Chen, Yichong Lu, Lanyun Zhu, Xiaowei Zhou, Andreas Geiger, Yiyi Liao\
> arXiv 2022

![image](https://fuxiao0719.github.io/projects/panopticnerf/images/pipeline.jpg)

## Installation
1. Create a virtual environment via `conda`.
    ```
    conda create -n panopticnerf python=3.7
    conda activate panopticnerf
    ```
2. Install `torch` and `torchvision`.
    ```
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```
3. Install requirements.
    ```
    pip install -r requirements.txt
    ```

## Data Preparation
1. We evaluate our model on [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/). Here we show the structure of a test dataset as follow. You can download it from [here](https://drive.google.com/file/d/1JkG8JbnVaxh3aMBo8vnKRT0Pvql1XQpU/view?usp=sharing) and then put it into `$ROOT` (RGBs should query the KITTI-360 website).
    ```
    ├── KITTI-360
      ├── 2013_05_28_drive_0000_sync
        ├── image_00
        ├── image_01
      ├── bbx_intersection
        ├── *_00.npz
        ├── *_01.npz
      ├── calibration
        ├── calib_cam_to_pose.txt
        ├── perspective.txt
      ├── data_3d_bboxes
      ├── data_poses
        ├── cam0_to_world.txt
        ├── poses.txt
      ├── pspnet
      ├── sgm
      ├── visible_id
    ```

    | file | Intro |
    | ------ | ------ |
    | `image_00/01` | stereo RGB images |
    | `pspnet` | 2D pseudo ground truth |
    | `sgm` | weak stereo depth supervision |
    | `visible_id` | per-frame bounding primitive IDs |
    | `data_poses` | system poses in a global Euclidean coordinate |
    | `calibration` | extrinsics and intrinsics of the perspective cameras |
    | `bbx_intersection` | ray-mesh intersections, containing depths between hitting points and camera origin, semantic label IDs and bounding primitive IDs|

2. Generate ray-mesh intersections (`bbx_intersection/*.npz`). The red dots and blue dots indicate where the rays hit into and out of the meshes, respectively. For the given test scene, `START=3353`, `NUM=64`.
    ```
    # image_00
    python mesh_intersection.py intersection_start_frame ${START} intersection_frames ${NUM} use_stereo False
    # image_01
    python mesh_intersection.py intersection_start_frame ${START} intersection_frames ${NUM} use_stereo True
    ```
  <img src="figs/ray_mesh_intersection.png" width="90%">

3. Evaluate the origin of a scene (`center_pose`) and the distance from the origin to the furthest bounding primitive (`dist_min`). Then accordingly modify the `.yaml` file.
    ```
    python recenter_pose.py recenter_start_frame ${START} recenter_frames ${NUM}
    ```

## Training and Visualization
1. We provide the training code. Replace `resume False` with `resume True` to load the pretained model.
    ```
    python train_net.py --cfg_file configs/panopticnerf_test.yaml pretrain nerf gpus '1,' use_stereo True use_pspnet True use_depth True pseudo_filter True weight_th 0.05 resume False
    ```

2. Render semantic map, panoptic map and depth map in a single forward pass, which takes around 10s per-frame on a single 3090 GPU. Please make sure to maximize the GPU memory utilization by increasing the size of the chunk to reduce inference time. Replace `use_stereo False` with `use_stereo True` to render the right views.
    ```
    python run.py --type visualize --cfg_file configs/panopticnerf_test.yaml use_stereo False
    ```
    <img src="figs/rendering.png" width="100%">
3. Visualize novel view appearance & label synthesis. Before rendering, select a frame and generate corresponding ray-mesh intersections with respect to its novel spiral poses by enabling `spiral poses==True` in `lib.datasets.kitti360.panopticnerf.py`. 

    ![monocular](figs/spiral.gif)

## Evaluation
  ```
  ├── KITTI-360
    ├── gt_2d_semantics
    ├── gt_2d_panoptics
    ├── lidar_depth
  ```
1. Download the corresponding [pretrained model](https://drive.google.com/drive/folders/1jd8eWfXDH7D09y4Ul1w7GKfTy3BqJZgk?usp=sharing) and put it to `$ROOT/data/trained_model/panopticnerf/panopticnerf_test/latest.pth`.

2. We provide some semantic & panoptic GTs and LiDAR point clouds for evaluation. The details of evaluation metrics can be found in the paper.
3. Eval mean intersection-over-union (mIoU)
  ```
  python run.py --type eval_miou --cfg_file configs/panopticnerf_test.yaml use_stereo False
  ```
4. Eval panoptic quality (PQ)
  ```
  sh eval_pq_test.sh
  ```
5. Eval depth with 0-100m LiDAR point clouds, where the far depth can be adjusted to evaluate the closer scene.
  ```
  python run.py --type eval_depth --cfg_file configs/panopticnerf_test.yaml use_stereo False max_depth 100.
  ```
6. Eval Multi-view Consistency (MC)
  ```
  python eval_consistency.py --cfg_file configs/panopticnerf_test.yaml use_stereo False consistency_thres 0.1
  ```

## News
* `12/04/2022` Code released.
* `29/03/2022` Repo created. Code will come soon.

## Citation

```bibtex
@article{fu2022panoptic,
  title={Panoptic NeRF: 3D-to-2D Label Transfer for Panoptic Urban Scene Segmentation},
  author={Fu, Xiao and Zhang, Shangzhan and Chen, Tianrun and Lu, Yichong and Zhu, Lanyun and Zhou, Xiaowei and Geiger, Andreas and Liao, Yiyi},
  journal={arXiv preprint arXiv:2203.15224},
  year={2022}
}
```
Copyright © 2022, Zhejiang University. All rights reserved. We favor any positive inquiry, please contact `lemonaddie0909@zju.edu.cn`.