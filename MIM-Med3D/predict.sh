#!/bin/bash
INPUT=$1
SAVE_PATH=$2

python ./code/experiments/sl/predict.py --config ./output/Fault_Finetune/swin_unetr_base_simmim500e_p16_public_256_flip_rotate_aug_4x4_rerun/config.yaml \
                                        --checkpoint ./output/Fault_Finetune/swin_unetr_base_simmim500e_p16_public_256_flip_rotate_aug_4x4_rerun/checkpoints/best.ckpt \
                                        --input $INPUT \
                                        --save_path $SAVE_PATH \
                                        --device cuda:0
