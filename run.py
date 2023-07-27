from lib.config import cfg, args
import numpy as np
import os
import copy

def run_eval_psnr():
    import glob
    import tqdm
    import os
    import cv2
    import imageio
    import torch
    frame_list = cfg.val_list
    
    # perspective
    psnr = 0
    for item in frame_list:
        pred_img_path = os.path.join(cfg.result_dir, '2/img'+str(item)+'_pred_img_00.png')
        pred_img = (np.array(imageio.imread(pred_img_path)) / 255.).astype(np.float32)
        gt_img_path = os.path.join(cfg.train_dataset.img_root, 'image_00/data_rect/000000'+str(item)+'.png') 
        gt_img = (np.array(imageio.imread(gt_img_path)) / 255.).astype(np.float32)
        gt_img = cv2.resize(gt_img, (704, 188), interpolation=cv2.INTER_AREA)
        color_loss = torch.nn.MSELoss(reduction='mean')(torch.from_numpy(pred_img), torch.from_numpy(gt_img))
        psnr += (-10. * torch.log(color_loss) / torch.log(torch.Tensor([10.]))).item() / len(frame_list)
    print('psnr : {0}'.format(psnr))
    with open(f'data/result_{cfg.exp_name}.txt', 'a+') as f:
        f.write('PSNR:{:.3f} '.format(psnr))

    # fisheye
    psnr = 0
    left_fisheye_grid = np.load('datasets/KITTI-360/fisheye/grid_fisheye_02.npy')
    left_fisheye_grid = left_fisheye_grid.reshape(1400, 1400, 4)[::4, ::4, :].reshape(-1, 4)
    mask_left = np.load('datasets/KITTI-360/fisheye/mask_left_fisheye.npy')[::4,::4]
    left_valid = ((left_fisheye_grid[:, 3] < 0.5) & (mask_left.reshape(-1) < 0.5)).reshape(350, 350)
    right_fisheye_grid = np.load('datasets/KITTI-360/fisheye/grid_fisheye_03.npy')
    right_fisheye_grid = right_fisheye_grid.reshape(1400, 1400, 4)[::4, ::4, :].reshape(-1, 4)
    mask_right = np.load('datasets/KITTI-360/fisheye/mask_right_fisheye.npy')[::4,::4]
    right_valid = ((right_fisheye_grid[:, 3] < 0.5) & (mask_right.reshape(-1) < 0.5)).reshape(350, 350)
    for item in frame_list:
        # image_02
        fisheye_img_path = os.path.join(cfg.result_dir, '2/img'+str(item)+'_fisheye_pred_img_00.png')
        fisheye_img = (np.array(imageio.imread(fisheye_img_path)) / 255.).astype(np.float32)
        gt_img = fisheye_img[:,350:,:]
        pred_img = fisheye_img[:,:350,:]
        color_loss = torch.nn.MSELoss(reduction='mean')(torch.from_numpy(pred_img[left_valid]), torch.from_numpy(gt_img[left_valid]))
        psnr += (-10. * torch.log(color_loss) / torch.log(torch.Tensor([10.]))).item() / len(frame_list) / 2 
        # image_03
        fisheye_img_path = os.path.join(cfg.result_dir, '2/img'+str(item)+'_fisheye_pred_img_01.png')
        fisheye_img = (np.array(imageio.imread(fisheye_img_path)) / 255.).astype(np.float32)
        gt_img = fisheye_img[:,350:,:]
        pred_img = fisheye_img[:,:350,:]
        color_loss = torch.nn.MSELoss(reduction='mean')(torch.from_numpy(pred_img[right_valid]), torch.from_numpy(gt_img[right_valid]))
        psnr += (-10. * torch.log(color_loss) / torch.log(torch.Tensor([10.]))).item() / len(frame_list) / 2 
    print('fisheye psnr : {0}'.format(psnr))
    with open(f'data/result_{cfg.exp_name}.txt', 'a+') as f:
        f.write('FE_PSNR:{:.3f} '.format(psnr))

def run_eval_miou():
    import glob
    import tqdm
    import os
    from lib.evaluators import make_evaluator
    evaluator = make_evaluator(cfg)
    val_list = cfg.val_list
    print(val_list)

    gt_list = []
    pred_list = []
    for frame in val_list:
        gt_list.append(os.path.join(cfg.semantic_gt_root, 'image_00', 'seq_000'+cfg.exp_name[19], 'semantic', '{:010d}.png'.format(frame)))
        pred_list.append(os.path.join(cfg.result_dir ,'2/img{}_pred_semantic_00.npy'.format(frame)))

    for idx in tqdm.tqdm(range(len(gt_list))):
        gt_path = gt_list[idx]
        pred_path = pred_list[idx]
        evaluator.evaluate(gt_path, pred_path)
    evaluator.summarize(is_fisheye=False)
    
def run_eval_miou_fe():
    import glob
    import tqdm
    import os
    from lib.evaluators import make_evaluator
    evaluator = make_evaluator(cfg)
    val_list = cfg.val_list
    val_list.remove(val_list[0])
    val_list.remove(val_list[-1])
    print(val_list)

    gt_list = []
    pred_list = []
    for frame in val_list:
        gt_list.append(os.path.join(cfg.semantic_gt_root, 'image_02', 'seq_000'+cfg.exp_name[19], 'semantic', '{:010d}.png'.format(frame)))
        pred_list.append(os.path.join(cfg.result_dir ,'2/img{}_fisheye_pred_semantic_00.npy'.format(frame)))
        gt_list.append(os.path.join(cfg.semantic_gt_root, 'image_03', 'seq_000'+cfg.exp_name[19], 'semantic', '{:010d}.png'.format(frame)))
        pred_list.append(os.path.join(cfg.result_dir ,'2/img{}_fisheye_pred_semantic_01.npy'.format(frame)))
        
    for idx in tqdm.tqdm(range(len(gt_list))):
        gt_path = gt_list[idx]
        pred_path = pred_list[idx]
        evaluator.evaluate(gt_path, pred_path)
    evaluator.summarize(is_fisheye=True)

def run_process_json():
    from tools.process_json import ProcessJson, ProcessJson_Fisheye
    import json
    pj = ProcessJson()
    pj.process(cfg.start, cfg.start+cfg.test_frames)
    with open(os.path.join(cfg.result_dir,'pred.json'), 'w') as fp:
        json.dump(pj._predictions, fp)
    pj_fe = ProcessJson_Fisheye()
    pj_fe.process(cfg.start, cfg.start+cfg.test_frames)
    with open(os.path.join(cfg.result_dir,'pred_fe.json'), 'w') as fp:
        json.dump(pj_fe._predictions, fp)

def run_process_json_crf():
    from tools.process_json import ProcessJsonCRF, ProcessJsonCRF_Fisheye
    import json
    pj_fe = ProcessJsonCRF()
    pj_fe.process(cfg.start, cfg.start+cfg.test_frames)
    with open(os.path.join(cfg.result_dir,'pred_crf.json'), 'w') as fp:
        json.dump(pj_fe._predictions, fp)
    pj_fe = ProcessJsonCRF_Fisheye()
    pj_fe.process(cfg.start, cfg.start+cfg.test_frames)
    with open(os.path.join(cfg.result_dir,'pred_fe_crf.json'), 'w') as fp:
        json.dump(pj_fe._predictions, fp)

def run_process_json_gt():
    from tools.process_json import ProcessJsonGT, ProcessJsonGT_Fisheye
    import json
    pj = ProcessJsonGT()
    pj.process(cfg.start, cfg.start+cfg.test_frames)
    with open(os.path.join(cfg.result_dir,'gt.json'), 'w') as fp:
        json.dump(pj._predictions, fp)
    pj_fe = ProcessJsonGT_Fisheye()
    pj_fe.process(cfg.start, cfg.start+cfg.test_frames)
    with open(os.path.join(cfg.result_dir,'gt_fe.json'), 'w') as fp:
        json.dump(pj_fe._predictions, fp)   

def run_eval_pq():
    from lib.evaluators.eval_pq import evaluate
    evaluate()

def run_eval_pq_fe():
    from lib.evaluators.eval_pq_fe import evaluate
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
        pred_depth = np.load(cfg.result_dir+'/2/img'+str(frame)+'_pred_depth_00.npy')
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_AREA)
        gt_all = np.concatenate((gt_all, gt_depth[mask]),axis=0)
        pred_all = np.concatenate((pred_all, pred_depth[mask]),axis=0)
    abs_rel, sq_rel, rmse, rmse_log, a1 = compute_errors(gt_all, pred_all)
    print('rmse:{0}'.format(rmse))
    print('a1:{0}'.format(a1))
    with open(f'data/result_{cfg.exp_name}.txt', 'a+') as f:
        f.write('RMSE:{:.3f} A1:{:.3f} '.format(rmse, a1))
        f.write('\n')

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
    network = imp.load_source(cfg.network_module, cfg.network_path).Network().cuda()
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
        torch.cuda.empty_cache()

if __name__ == '__main__':
    print(args.type)
    globals()['run_' + args.type]()
