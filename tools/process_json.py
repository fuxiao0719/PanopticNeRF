import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from tabulate import tabulate
import glob
from lib.config import cfg, args
import cv2
logger = logging.getLogger(__name__)
eval_instance_list = [11, 26, 29, 30, 41]

class ProcessJson():
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`
    It contains a synchronize call and has to be called from all workers.
    """
    def __init__(self):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._predictions = []

    def process(self, start = -1, end = -1):
        store_dir = os.path.join(cfg.result_dir)
        id2semantic_paths = sorted(glob.glob(os.path.join(store_dir, '2/img*_id2semantic_00.npy')))
        panoptic_paths = sorted(glob.glob(os.path.join(store_dir, '2/img*_panoptic_id_map_00.npy')))
        for id2semantic_path, panoptic_path, in zip(id2semantic_paths, panoptic_paths):
            file_name = os.path.basename(panoptic_path)
            if file_name[8:15] == 'fisheye':
                continue
            image_id = int(file_name[3:7])
            if (start != -1) and (end != -1):
                if (image_id > end) or (image_id < start):
                    break
            if cfg.exp_name[19] == '0':
                seq = 'seq_0000'
            else:
                seq = 'seq_0004'
            panoptic_gt_path = os.path.join(cfg.panoptic_gt_root,'image_00', seq, 'instance', "{:010}.png".format(image_id))
            if os.path.exists(panoptic_gt_path) != True:
                continue
            panoptic_gt = cv2.imread(panoptic_gt_path, -1)
            panoptic_img = np.load(panoptic_path, allow_pickle=False)
            panoptic_gt = cv2.resize(panoptic_gt, (panoptic_gt.shape[1]//2, panoptic_gt.shape[0]//2), cv2.INTER_NEAREST)
            id2semantic = np.load(id2semantic_path, allow_pickle=True)
            id2semantic =  id2semantic.item()
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == 0:
                    # VOID region.
                    continue
                if panoptic_label < 50:
                    pred_class = panoptic_label
                else:
                    pred_class = id2semantic[panoptic_label]
                if pred_class == 39:
                    pred_class = 41
                if pred_class == 34:
                    pred_class = 11
                isthing =   (
                    int(pred_class)  in eval_instance_list
                )
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                    }
                )
            # Official evaluation script uses 0 for VOID label.
            panoptic_img += 1
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                    "path": panoptic_path,
                }
            )


class ProcessJson_Fisheye():
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`
    It contains a synchronize call and has to be called from all workers.
    """
    def __init__(self):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._predictions = []

    def process(self, start = -1, end = -1):
        store_dir = os.path.join(cfg.result_dir)
        id2semantic_paths = sorted(glob.glob(os.path.join(store_dir, '2/img*_fisheye_id2semantic_00.npy')))
        panoptic_paths = sorted(glob.glob(os.path.join(store_dir, '2/img*_fisheye_panoptic_id_map_00.npy')))
        id2semantic_paths += sorted(glob.glob(os.path.join(store_dir, '2/img*_fisheye_id2semantic_01.npy')))
        panoptic_paths += sorted(glob.glob(os.path.join(store_dir, '2/img*_fisheye_panoptic_id_map_01.npy')))
        for id2semantic_path, panoptic_path, in zip(id2semantic_paths, panoptic_paths):
            file_name = os.path.basename(panoptic_path)
            image_id = int(file_name[3:7])
            if (start != -1) and (end != -1):
                if (image_id > end) or (image_id < start):
                    break
            if cfg.exp_name[19] == '0':
                seq = 'seq_0000'
            else:
                seq = 'seq_0004'
            if id2semantic_path[-5] == '0':
                image_dir = 'image_02'
            else:
                image_dir = 'image_03'
            panoptic_gt_path = os.path.join(cfg.panoptic_gt_root, image_dir, seq, 'instance', "{:010}.png".format(image_id))
            if os.path.exists(panoptic_gt_path) != True:
                continue
            panoptic_gt = cv2.imread(panoptic_gt_path, -1)
            panoptic_img = np.load(panoptic_path, allow_pickle=False)
            panoptic_gt = cv2.resize(panoptic_gt, (panoptic_gt.shape[1]//2, panoptic_gt.shape[0]//2), cv2.INTER_NEAREST)
            id2semantic = np.load(id2semantic_path, allow_pickle=True)
            id2semantic =  id2semantic.item()
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == 0:
                    # VOID region.
                    continue
                if panoptic_label < 50:
                    pred_class = panoptic_label
                else:
                    pred_class = id2semantic[panoptic_label]
                if pred_class == 39:
                    pred_class = 41
                if pred_class == 34:
                    pred_class = 11
                isthing =   (
                    int(pred_class)  in eval_instance_list
                )
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                    }
                )
            # Official evaluation script uses 0 for VOID label.
            panoptic_img += 1
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                    "path": panoptic_path,
                }
            )


class ProcessJsonGT():
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._predictions = []

    def process(self, start = -1, end = -1):
        if cfg.exp_name[19] == '0':
            seq = 'seq_0000'
        else:
            seq = 'seq_0004'
        panoptic_paths = sorted(glob.glob(os.path.join(cfg.panoptic_gt_root, 'image_00', seq, 'instance', '*.png')))
        for panoptic_path in panoptic_paths:
            file_name = os.path.basename(panoptic_path)
            image_id = int(file_name[-8:-4])
            if (start != -1) and (end != -1):
                if (image_id > end) or (image_id < start):
                    continue
            if os.path.exists(panoptic_path) != True:
                continue
            panoptic_img = cv2.imread(panoptic_path, -1)
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == 0:
                    # VOID region.
                    continue
                pred_class = int(panoptic_label / 1000)
                if pred_class == 39:
                    pred_class = 41
                if pred_class == 34:
                    pred_class = 11
                isthing = (
                    int(pred_class) in eval_instance_list
                )
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                        "iscrowd": 0
                    }
                )
            panoptic_img += 1
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                    "path": panoptic_path,
                }
            )


class ProcessJsonGT_Fisheye():
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._predictions = []

    def process(self, start = -1, end = -1):
        if cfg.exp_name[19] == '0':
            seq = 'seq_0000'
        else:
            seq = 'seq_0004'
        panoptic_paths = sorted(glob.glob(os.path.join(cfg.panoptic_gt_root,'image_02', seq, 'instance', '*.png')))
        panoptic_paths += sorted(glob.glob(os.path.join(cfg.panoptic_gt_root,'image_03', seq, 'instance', '*.png')))
        for panoptic_path in panoptic_paths:
            file_name = os.path.basename(panoptic_path)
            image_id = int(file_name[-8:-4])
            if (start != -1) and (end != -1):
                if (image_id > end) or (image_id < start):
                    continue
            if os.path.exists(panoptic_path) != True:
                continue
            panoptic_img = cv2.imread(panoptic_path, -1)
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == 0:
                    # VOID region.
                    continue
                pred_class = int(panoptic_label / 1000)
                if pred_class == 39:
                    pred_class = 41
                if pred_class == 34:
                    pred_class = 11
                isthing = (
                    int(pred_class) in eval_instance_list
                )
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                        "iscrowd": 0
                    }
                )
            panoptic_img += 1
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                    "path": panoptic_path,
                }
            )

class ProcessJsonCRF():
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """
    
    def __init__(self):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._predictions = []

    def process(self, start = -1, end = -1):
        if cfg.exp_name[19] == '0':
            seq = 'seq_0000'
        else:
            seq = 'seq_0004'
        panoptic_paths = sorted(glob.glob(os.path.join('datasets/KITTI-360/baselines/crf/instance', seq, '*.png')))
        panoptic_paths += sorted(glob.glob(os.path.join('datasets/KITTI-360/baselines/crf/instance', seq, '*.png')))
        for panoptic_path in panoptic_paths:
            file_name = os.path.basename(panoptic_path)
            image_id = int(file_name[-8:-4])
            if (start != -1) and (end != -1):
                if (image_id > end) or (image_id < start):
                    continue
            if os.path.exists(panoptic_path) != True:
                continue
            panoptic_img = cv2.imread(panoptic_path, -1)
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == 0:
                    # VOID region.
                    continue
                pred_class = int(panoptic_label / 1000)
                if pred_class == 39:
                    pred_class = 41
                if pred_class == 34:
                    pred_class = 11
                isthing = (
                    int(pred_class) in eval_instance_list
                )
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                        "iscrowd": 0
                    }
                )
            panoptic_img += 1
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                    "path": panoptic_path,
                }
            )

class ProcessJsonCRF_Fisheye():
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """
    
    def __init__(self):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._predictions = []

    def process(self, start = -1, end = -1):
        if cfg.exp_name[19] == '0':
            seq = 'seq_0000'
        else:
            seq = 'seq_0004'
        panoptic_paths = sorted(glob.glob(os.path.join('datasets/KITTI-360/baselines/crf/instance_fe/image_02', seq, '*.png')))
        panoptic_paths += sorted(glob.glob(os.path.join('datasets/KITTI-360/baselines/crf/instance_fe/image_03', seq, '*.png')))
        for panoptic_path in panoptic_paths:
            file_name = os.path.basename(panoptic_path)
            image_id = int(file_name[-8:-4])
            if (start != -1) and (end != -1):
                if (image_id > end) or (image_id < start):
                    continue
            if os.path.exists(panoptic_path) != True:
                continue
            panoptic_img = cv2.imread(panoptic_path, -1)
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == 0:
                    # VOID region.
                    continue
                pred_class = int(panoptic_label / 1000)
                if pred_class == 39:
                    pred_class = 41
                if pred_class == 34:
                    pred_class = 11
                isthing = (
                    int(pred_class) in eval_instance_list
                )
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                        "iscrowd": 0
                    }
                )
            panoptic_img += 1
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": segments_info,
                    "path": panoptic_path,
                }
            )