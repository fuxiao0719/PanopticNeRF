import os
import cv2
import json
import time
import numpy as np
import PIL.Image as Image
from lib.config import cfg
from collections import OrderedDict
from collections import defaultdict
from tabulate import tabulate
import multiprocessing
from tools.kitti360scripts.helpers.labels import labels
import glob

eval_instance_list = [11, 26, 29, 30, 41]
categories_global = [{"supercategory": label.name, "id": label.id, "name": label.name, 'isthing': 1 if label.id in eval_instance_list  else 0} for label in labels]
id2name = {label.id: label.name  for label in labels}
OFFSET = 256 * 256 * 256
VOID = 0

# load fisheye grids
left_fisheye_grid = np.load('datasets/KITTI-360/fisheye/grid_fisheye_02.npy')
left_fisheye_grid = left_fisheye_grid.reshape(1400, 1400, 4)[::4, ::4, :].reshape(-1, 4)
mask_left = np.load('datasets/KITTI-360/fisheye/mask_left_fisheye.npy')[::4, ::4]
left_valid = ((left_fisheye_grid[:, 3] < 0.5) & (mask_left.reshape(-1) < 0.5)).reshape(350, 350)

right_fisheye_grid = np.load('datasets/KITTI-360/fisheye/grid_fisheye_03.npy')
right_fisheye_grid = right_fisheye_grid.reshape(1400, 1400, 4)[::4, ::4, :].reshape(-1, 4)
mask_right = np.load('datasets/KITTI-360/fisheye/mask_right_fisheye.npy')[::4, ::4]
right_valid = ((right_fisheye_grid[:, 3] < 0.5) & (mask_right.reshape(-1) < 0.5)).reshape(350, 350)

class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self

class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)
    
    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            if (label == 17) or (label == 20) or (label == 38):
                continue
            
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            if pq_class + sq_class + rq_class == 0:
                continue
            n += 1
            pq += pq_class
            sq += sq_class
            rq += rq_class
        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results
    
def filter_gt(pan_gt, gt_ann):
    labels, labels_cnt = np.unique(pan_gt, return_counts=True)
    remove_id = []
    for label, label_cnt in zip(labels, labels_cnt):
        if (label == 6000) or (label == 17000) or (label == 20000) or (label == 38000) or (int(label/1000) == 33) or (label_cnt < 100):
            pan_gt[pan_gt == label] = VOID
            remove_id.append(label)
    return pan_gt, remove_id

def filter_pred(pan_pred, pred_ann):
    labels, labels_cnt = np.unique(pan_pred, return_counts=True)
    remove_id = []
    for label, label_cnt in zip(labels, labels_cnt):
        if (label == 0) or (label == 6) or (label == 17) or (label == 20) or (label == 33) or (label == 38) or (label == 50):
            pan_pred[pan_pred == label] = VOID
            remove_id.append(label)
        elif (label_cnt < 100):
            pan_pred[pan_pred == label] = 23
            remove_id.append(label)
    return pan_pred, remove_id

def pq_compute_single_core(proc_id, annotation_set, categories):
    pq_stat = PQStat()
    match_set = {}
    idx = 0
    for gt_ann, pred_ann in annotation_set:
        idx += 1
        pan_gt = np.array(Image.open(gt_ann['path']), dtype=np.uint32)
        if pred_ann['path'][-5] == '0':
            valid = left_valid
        else:
            valid = right_valid
        try:
            pan_pred = np.zeros((350,350), dtype=np.int64)
            pan_pred_valid = np.load(pred_ann['path'])
            pan_pred[valid] = pan_pred_valid
        except:
            pan_pred = cv2.imread(pred_ann['path'],-1)
        pan_pred = cv2.resize(pan_pred, (pan_gt.shape[1], pan_gt.shape[0]), interpolation = cv2.INTER_NEAREST)
        pan_pred = pan_pred.astype(np.uint32)
        pan_gt, gt_remove_id = filter_gt(pan_gt, gt_ann)
        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        for i in gt_remove_id:
            if i == 0:
                continue
            gt_segms.pop(i)

        pan_pred, pred_remove_id = filter_pred(pan_pred, pred_ann)
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}
        for i in pred_remove_id:
            if i == 0:
                continue
            pred_segms.pop(i)
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:
                continue
            semantic_label = pred_segms[label]['category_id']
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))

        gt_labels_set = set(el['id'] for el in gt_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_gt, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:
                continue
            gt_segms[label]['area'] = label_cnt
            gt_labels_set.remove(label)
        
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection
        
        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        matched_dict = {}
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue
            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
                matched_dict[gt_label] = pred_label
        match_set[pred_ann['file_name']] = matched_dict

        # count false negative
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positive
        for pred_label, pred_info in pred_segms.items():
            semantic_label = pred_info['category_id']
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] == 6:
                continue
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    return pq_stat, match_set

def pq_compute_multi_core(matched_annotations_list, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    match_set = {}
    for proc_id, annotation_set in enumerate(annotations_split):
        debug = True
        if debug == False:
            p = workers.apply_async(pq_compute_single_core, (proc_id, annotation_set, categories))
            p, match = p.get()
            processes.append(p)
            match_set.update(match)
        else:
            p, match = pq_compute_single_core(proc_id, annotation_set, categories)
            processes.append(p)
            match_set.update(match)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p
    return pq_stat, match_set

def pq_compute(gt_json_file, pred_json_file):
    start_time = time.time()
    gt_json = gt_json_file
    pred_json = pred_json_file
    categories = {el['id']: el for el in categories_global}

    # print("Evaluation panoptic segmentation metrics:")
    matched_annotations_list = []
    for i in range(len(gt_json)):
        # raise Exception('no prediction for the image with id: {}'.format(image_id))
        assert gt_json[i]['image_id'] == pred_json[i]['image_id']
        matched_annotations_list.append((gt_json[i], pred_json[i]))

    pq_stat, match_set = pq_compute_multi_core(matched_annotations_list, categories)
    with open(os.path.join(cfg.result_dir,'match_set_fe.json'), 'w') as fp:
        json.dump(match_set, fp)
    
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
            with open(f'data/result_{cfg.exp_name}.txt', 'a+') as f:
                f.write('FE_PQ:{:.3f} FE_SQ:{:.3f} FE_RQ:{:.3f} N:{:.0f} '.format(results[name]['pq'], results[name]['sq'], results[name]['rq'], results[name]['n']))

    for id_, res_dict in results['per_class'].items():
        pq = res_dict['pq']
        sq = res_dict['sq']
        rq = res_dict['rq']
        if pq+sq+rq != 0:
            print("{}: pq {:.3f}, sq {:.3f}, rq {:.3f}".format(id2name[id_], pq, sq, rq))

    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))
    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )
    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))
    return results

def evaluate():

    path_list = glob.glob(cfg.result_dir)

    gt_json = []
    pred_json = []

    for path in path_list:
        with open(os.path.join(path, 'gt_fe.json'), 'r') as f:
            gt_json_per = json.load(f)
        with open(os.path.join(path, 'pred_fe.json'), 'r') as f:
            pred_json_per = json.load(f)
        gt_json += gt_json_per
        pred_json += pred_json_per

    pq_res = pq_compute(
        gt_json,
        pred_json,
    )

    res = {}
    res["PQ"] = 100 * pq_res["All"]["pq"]
    res["SQ"] = 100 * pq_res["All"]["sq"]
    res["RQ"] = 100 * pq_res["All"]["rq"]
    res["PQ_th"] = 100 * pq_res["Things"]["pq"]
    res["SQ_th"] = 100 * pq_res["Things"]["sq"]
    res["RQ_th"] = 100 * pq_res["Things"]["rq"]
    res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
    res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
    res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

    results = OrderedDict({"panoptic_seg": res})
    _print_panoptic_results(pq_res)
    return results

def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
