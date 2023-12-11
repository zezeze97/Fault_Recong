#!/bin/bash
INPUT=$1
SAVE_PATH=$2

python ./code/experiments/sl/predict.py --config ./output/Fault_Finetuning/swin_unetr_base_simmim500e_p16_public_whole_random_crop_1x4/config.yaml \
                                        --checkpoint ./output/Fault_Finetuning/swin_unetr_base_simmim500e_p16_public_whole_random_crop_1x4/checkpoints/best.ckpt \
                                        --input $INPUT \
                                        --save_path $SAVE_PATH \
                                        --device cuda:0
