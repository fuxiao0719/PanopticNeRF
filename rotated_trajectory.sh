local_rank=0
frame_num=64
scene=panopticnerf360_test
spiral_frame=1930
python mesh_intersection_rotated_trajectory.py intersection_spiral_frame ${spiral_frame} intersection_frames ${frame_num} use_stereo False
python run.py --type visualize --cfg_file configs/${scene}_spiral.yaml use_stereo False use_post_processing False gpus "${local_rank}," intersection_spiral_frame ${spiral_frame} intersection_frames ${frame_num}
