#!/bin/bash
GPU=$2
port=23509


config=swin-base-patch4-window7_upernet_8xb2-160k_mix_data_force_3_chan-512x512_per_image_normal_no_pretrain
if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PORT=${port} ./tools/dist_train.sh ./projects/Fault_recong/config/${config}.py 8 --work-dir output/${config}
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128.py ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/iter_48000.pth 1
fi