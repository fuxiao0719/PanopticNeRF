local_rank=0
epochs=5
scene=panopticnerf360_seq0_6398_6461_init
echo ${local_rank}
echo ${scene}
echo ${epochs}
rm -rf data/trained_model/panopticnerf360/${scene}_ft
mkdir data/trained_model/panopticnerf360/${scene}_ft
cp -r data/trained_model/panopticnerf360/${scene}/latest.pth data/trained_model/panopticnerf360/${scene}_ft/latest.pth
python run.py --type visualize --cfg_file configs/${scene}.yaml exp_name "${scene}_ft" use_stereo True use_post_processing True gpus "${local_rank},"  
python train_net.py --cfg_file configs/"${scene}".yaml exp_name "${scene}_ft" use_stereo True use_pspnet True use_depth True use_pseudo_filter True weight_th 0.05 use_instance_loss True train.epoch ${epochs} resume True gpus "${local_rank},"
