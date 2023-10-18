#!/bin/bash

CONFIG=$1
CKPTS=$2
SAVE_ROOT_PATH=$3

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /home/zhangzr/FaultRecongnition/Fault_data/project_data_v3/guai3east/seis.sgy \
                                        --save_path $SAVE_ROOT_PATH/guai3east_pred/ \
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:0 \
                                        --direction xline &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /home/zhangzr/FaultRecongnition/Fault_data/project_data_v3/guai3east/seis.sgy \
                                        --save_path $SAVE_ROOT_PATH/guai3east_pred/ \
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:1 \
                                        --direction inline &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /home/zhangzr/FaultRecongnition/Fault_data/project_data_v3/madonglianpian/seis.sgy \
                                        --save_path $SAVE_ROOT_PATH/madonglianpian_pred/ \
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:2 \
                                        --direction xline &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /home/zhangzr/FaultRecongnition/Fault_data/project_data_v3/madonglianpian/seis.sgy \
                                        --save_path $SAVE_ROOT_PATH/madonglianpian_pred/ \
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:3 \
                                        --direction inline &