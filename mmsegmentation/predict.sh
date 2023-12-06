#!/bin/bash
INPUT=$1
SAVEPATH=$2


python ./projects/Fault_recong/predict.py --config ./output/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e.py \
                                        --checkpoint ./output/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e/best.pth \
                                        --input $INPUT \
                                        --save_path $SAVEPATH \
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:0 \
                                        --direction inline 