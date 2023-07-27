local_rank=0
scene=panopticnerf360_test
frame=1947
x_theta=90.
y_theta=140.
z_theta=0.
echo ${frame}
echo ${x_theta}
echo ${y_theta}
echo ${z_theta}
python mesh_intersection_360.py intersection_spiral_frame ${frame}  x_theta ${x_theta} y_theta ${y_theta} z_theta ${z_theta} 
python run.py --type visualize --cfg_file configs/${scene}_360.yaml use_stereo False gpus "${local_rank}," intersection_spiral_frame ${frame} x_theta ${x_theta} y_theta ${y_theta} z_theta ${z_theta} 
