import cv2
import numpy as np
from tools.kitti360scripts.helpers.labels import labels, id2label
from collections import defaultdict
from lib.config import cfg, args
import os
import torch

def fast_hist(label_true, label_pred, n_classes):
    mask = (label_true >= 0) & (label_true < n_classes)
    hist = np.bincount(
        n_classes * label_true[mask] + label_pred[mask],
        minlength = n_classes ** 2,
    ).reshape(n_classes, n_classes)
    return hist

def per_class_iu(hist):
    return np.diag(hist) / (1e-6+(hist.sum(1) + hist.sum(0) - np.diag(hist)))

def compute_mIoU(pred, label, n_classes):
    hist = np.zeros((n_classes, n_classes))
    hist += fast_hist(label.flatten(), pred.flatten(), n_classes)
    mIoUs = per_class_iu(hist)
    return mIoUs

label = [label.id for label in labels]
label2name = {label.id: label.name for label in labels}
color2id = {}
def merge(pred):
    pred[pred==39]=41
    return pred

class Evaluator:
    def __init__(self,):
        self.frames = []
        self.miou_dict = {}
        self.label_dict = defaultdict(list)
        self.gt_list = []
        self.pred_list = []
    
    def evaluate(self, gt_path, pred_path, test_set):
        frame = int(gt_path[-8:-4])
        self.frames.append(frame)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.int)
        pred = np.load(pred_path).astype(np.int)[..., 0]
        pred_upscale = cv2.resize(pred, (gt.shape[1],gt.shape[0]), interpolation = cv2.INTER_NEAREST)
        pred = pred_upscale
        mask = (gt!=255) & (gt!=0) & (pred!=255) & (pred!=0) &(gt!=38) & (gt!=17)
        gt = gt[mask]
        pred = pred[mask]
        pred = merge(pred)
        self.gt_list.append(gt)
        self.pred_list.append(pred)
        mIoUs = compute_mIoU(pred, gt, len(label))
        for i in range(len(mIoUs)):
            if mIoUs[i] > 0 and mIoUs[i] < 1:
                self.label_dict[label2name[label[i]]].append(mIoUs[i])
        mask = (mIoUs>0) & (mIoUs<1)
        mIoUs = mIoUs[mask]
        self.miou_dict[frame] = np.mean(mIoUs)
    
    def summarize(self):
        miou_list = []
        gt_all = np.concatenate(self.gt_list)
        pred_all = np.concatenate(self.pred_list)
        label_all_dict = defaultdict(list)
        mIoUs = compute_mIoU(pred_all, gt_all, len(label))
        for i in range(len(mIoUs)):
            if mIoUs[i] > 0.1 and mIoUs[i] < 1:
                label_all_dict[label2name[label[i]]].append(mIoUs[i])
        mask = (mIoUs>0.1) & (mIoUs<1)
        mIoUs = mIoUs[mask]
        for item in label_all_dict.values():
            miou_list.append(item)
        print('miou: {}'.format(np.mean(np.array(mIoUs))))
        for i in label_all_dict.keys():
            print('IoU of {0} is {1}'.format(i, np.mean(label_all_dict[i])))
        print('total acc:{}'.format((pred_all==gt_all).sum()/len(gt_all)))