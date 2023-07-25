#!/bin/bash

PRED_PATH=$1

python evalFault.py --gt_path /home/zhangzr/Fault_Recong/Fault_data/real_labeled_data/origin_data/fault/label_fill.sgy \
                    --pred_path $PRED_PATH \
                    --scalefactor 3 \
                    --step 1 \
                    --start_idx 373