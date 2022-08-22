from lib.config import cfg, args
import numpy as np
import os

def run_eval_miou():
    import glob
    import tqdm
    import os
    from lib.evaluators import make_evaluator
    evaluator = make_evaluator(cfg)
    val_list = cfg.val_list
    print(val_list)
    gt_list = []
    for item in val_list:
        gt_list.append(os.path.join(cfg.semantic_gt_root, '000000'+str(item)+'.png'))
    for gt_path in tqdm.tqdm(gt_list):
        frame = int(gt_path[-8:-4])
        if (frame < cfg.start) or (frame > cfg.start + cfg.train_frames):
            continue
        pred_path = os.path.join(cfg.result_dir ,'2/img{}_pred_semantic.npy'.format(frame))
        evaluator.evaluate(gt_path, pred_path, '')
    evaluator.summarize()

def run_process_json():
    from tools.process_json import ProcessJson
    import json
    pj = ProcessJson()
    pj.process()
    with open(os.path.join(cfg.result_dir,'pred.json'), 'w') as fp:
        json.dump(pj._predictions, fp)

def run_process_json_gt():
    from tools.process_json import ProcessJsonGT
    import json
    pj = ProcessJsonGT()
    pj.process(cfg.start, cfg.start+cfg.test_frames)
    with open(os.path.join(cfg.result_dir,'gt.json'), 'w') as fp:
        json.dump(pj._predictions, fp)

def run_eval_pq():
    from lib.evaluators.eval_pq import evaluate
    evaluate()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25 ).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1

def run_eval_depth():
    import glob
    import tqdm
    import os
    import cv2
    frame_list = cfg.val_list
    # evaluate on all views
    gt_list = []
    pred_all = np.array([])
    gt_all = np.array([])
    for item in frame_list:
        gt_list.append(os.path.join(cfg.lidar_depth_root, '000000'+str(item)+'_0.npy'))
    for gt_path in tqdm.tqdm(gt_list):
        gt_depth = np.load(gt_path)
        mask = (gt_depth>0) & (gt_depth<cfg.max_depth)
        frame = int(gt_path[-10:-6])
        pred_depth = np.load(cfg.result_dir+'/2/img'+str(frame)+'_depth.npy')
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_AREA)
        gt_all = np.concatenate((gt_all, gt_depth[mask]),axis=0)
        pred_all = np.concatenate((pred_all, pred_depth[mask]),axis=0)
    abs_rel, sq_rel, rmse, rmse_log, a1 = compute_errors(gt_all, pred_all)
    print('rmse:{0}'.format(rmse))
    print('a1:{0}'.format(a1))

def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network, load_pretrain
    from lib.utils import net_utils
    import tqdm
    import torch
    import imp
    from lib.visualizers import make_visualizer
    from lib.utils.data_utils import to_cuda
    network = imp.load_source(cfg.network_module, cfg.network_path).Network().cuda(cfg.local_rank)
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    load_pretrain(network, cfg.exp_name)
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch)
            visualizer.visualize(output, batch)

if __name__ == '__main__':
    print(args.type)
    globals()['run_' + args.type]()
