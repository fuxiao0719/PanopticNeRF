from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

cfg = CN()

# task settings
cfg.eval_mode = 0
cfg.input_sparse = False
cfg.combine_3d_2d = 1
cfg.save_img = False
cfg.N_rays = 2048
cfg.ft_scene = ''
cfg.eval_setting = 'enerf' # ['mvsnerf', 'enerf']
cfg.depth_inv = True
cfg.render_scale = 1.0
cfg.train_frames = 100
cfg.test_frames = 5
cfg.recenter_start_frame = 3353
cfg.recenter_frames = 64
cfg.intersection_start_frame = 3353
cfg.intersection_spiral_frame = 1568
cfg.intersection_frames = 64
cfg.spiral_frame = 3400
cfg.spiral_frame_num = 32
cfg.render_cam = -1
cfg.sample_more_onmask = False
cfg.use_pspnet = True
cfg.enhance_3d = False
cfg.val_list = []
cfg.depth_object = []
cfg.start = 6060
cfg.decay_rate = 1.
cfg.lambda_depth = 0.1
cfg.bbox_sp = 10
cfg.mode = 0
cfg.mask_parking = True
cfg.semantic_weight = 0.0005
cfg.weight_th = 0.
cfg.dist_lambda = 0.005
cfg.xyz_res = 6
cfg.view_res = 4
cfg.ratio = 0.5
cfg.lambda_fix = 1.
cfg.lambda_semantic_2d = 1.
cfg.lambda_3d = 1.
cfg.crf_seq = -1
cfg.consistency_thres = -1.
cfg.pseudo_filter = False
cfg.use_decay = True
cfg.train_baseline1 = False
cfg.only_baseline2 = False
cfg.lidar_frames = 1
cfg.samples_all = 192
cfg.vis_index = 0
cfg.vis_x = 298
cfg.vis_y = 97
cfg.vis_depth = 1
cfg.center_pose = []
cfg.cam_interval = 1
cfg.log_sample = False
cfg.use_pspnet = False
cfg.postprocessing = False
cfg.max_depth = -1.
cfg.lidar_samples = 64
cfg.detach = True
cfg.use_depth = True
cfg.use_stereo = True
cfg.dist = 300
cfg.sampling_change = False
cfg.test_start = 7300
cfg.iterative_train = False
cfg.render_instance = False
# cfg.init_network = False
cfg.init_name = 'None'
cfg.trained_model_dir_init = 'data/trained_model'
cfg.lidar_depth_root = ''
cfg.semantic_gt_root = ''
cfg.panoptic_gt_root = ''

# module
cfg.train_dataset_module = 'lib.datasets.dtu.neus'
cfg.test_dataset_module = 'lib.datasets.dtu.neus'
cfg.val_dataset_module = 'lib.datasets.dtu.neus'
cfg.network_module = 'lib.neworks.neus.neus'
cfg.loss_module = 'lib.train.losses.neus'
cfg.evaluator_module = 'lib.evaluators.neus'
cfg.test_start = -1
# experiment name
cfg.exp_name = 'gittag_hello'
cfg.pretrain = ''

# network
cfg.distributed = False

# task
cfg.task = 'hello'

# gpus
cfg.gpus = list(range(4))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 1
cfg.save_latest_ep = 1
cfg.eval_ep = 1
log_interval : 20

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------

cfg.train = CN()
cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True
cfg.train.weight_depth = 0.01
cfg.train.weight_line_of_sight = 10.0
cfg.train.weight_color = 1.

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.
cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4
cfg.train.acti_func = 'relu'
cfg.frozen = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.val_dataset = ''
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.epoch = -1
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})

# trained model
cfg.trained_model_dir = 'data/trained_model'
cfg.trained_config_dir = 'data/trained_config'

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False
cfg.fix_random = False

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')
    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.exp_name = cfg.exp_name.replace('gittag', os.popen('git describe --tags --always').readline().strip())
    cfg.trained_model_dir_init = os.path.join(cfg.trained_model_dir, cfg.task, cfg.init_name)
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)    
    cfg.trained_config_dir = os.path.join(cfg.trained_config_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.local_rank = args.local_rank
    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'

def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
