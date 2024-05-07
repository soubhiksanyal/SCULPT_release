#!/bin/bash
python train.py \
--outdir=./output \
--cfg=stylegan3-t \
--data=./data/RGB_with_same_pose_white_16362_withclothinglabel_withnormals_withcolors_MODNetSegment_withalpha.zip \
--gpus=8 --batch=32 --gamma=2 --snap=40 --aug=ada --workers 8 --glr 0.001 --dlr 0.001 \
--pose_cond=0 --pose_cond_type=axisangle --disp_activatn_type=tanh \
--disp_scale=1.0 --mesh_smooth_w 0.0 --blur_radius 0. --only_disp_img=0 --mask_disp_map=0 --resume_pretrain_cape=0 --tick 4 --faces_per_pixel 1 \
--seperate_disp_map=0 --sep_disp_map_sampling=0 --spiral_conv=0 \
--texture_render=1 --clothtype_cond=1 --conformnet=1 --conditional_d=0 --guass_blur_normals=0 --patch_d=1 --colorcond=1 \
--desc _reproduce_camreadyrelease \
--geometry=./data/network-snapshot-013440.pkl
# --resume=./data/network-snapshot-025000.pkl