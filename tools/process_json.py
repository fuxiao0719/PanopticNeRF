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
eval_instance_list = [11, 26, 30, 29, 41]

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
        instance2semantics = sorted(glob.glob(os.path.join(store_dir, '2/img*_instance2semantic.npy')))
        panoptic_img = sorted(glob.glob(os.path.join(store_dir, '2/img*_panoptic_id_map.npy')))
        for instance2semantic_path, panoptic_path, in zip(instance2semantics, panoptic_img):
            file_name = os.path.basename(panoptic_path)
            image_id = int(file_name[3:7])
            if (start != -1) and (end != -1):
                if (image_id > end) or (image_id < start):
                    break
            panoptic_gt_path = os.path.join(cfg.panoptic_gt_root, "{:010}.png".format(image_id))
            if os.path.exists(panoptic_gt_path) != True:
                continue
            panoptic_gt = cv2.imread(panoptic_gt_path, -1)
            panoptic_img = np.load(panoptic_path, allow_pickle=False)
            panoptic_gt = cv2.resize(panoptic_gt, (panoptic_gt.shape[1]//2, panoptic_gt.shape[0]//2), cv2.INTER_NEAREST)
            instance2semantic = np.load(instance2semantic_path, allow_pickle=True)
            instance2semantic =  instance2semantic.item()
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == 0:
                    # VOID region.
                    continue
                if panoptic_label < 50:
                    pred_class = panoptic_label
                else:
                    pred_class = instance2semantic[panoptic_label]
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
        panoptic_img = sorted(glob.glob(os.path.join(cfg.panoptic_gt_root, '*.png')))
        for panoptic_path in panoptic_img:
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
                }
            )
