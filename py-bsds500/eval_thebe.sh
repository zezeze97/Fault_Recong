#!/bin/bash

PRED_PATH=$1

python evalFault.py --gt_path ../Fault_data/public_data/precessed/test/fault/faulttest.npy \
                    --pred_path $PRED_PATH \
                    --step 5 \
                    --scalefactor 3 \
                    --UPPER_BOUND 1300 \
                    --LOWER_BOUND 800 \