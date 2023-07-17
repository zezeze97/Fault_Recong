#!/bin/bash

CONFIG=$1
CKPTS=$2
SAVE_ROOT_PATH=$3

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/labeled/Ordos/gjb/seis/L500_1500_T500_2000_aa_pstm_0922_cg.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/labeled/Ordos/gjb/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:0 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/labeled/Ordos/pl/seis/20230419_PLB-YW-pstm-post-yanshou-Q_biaoqian.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/labeled/Ordos/pl/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:1 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/labeled/Ordos/yw/seis/mig.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/labeled/Ordos/yw/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:2 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/labeled/qyb/seis/20230412_QY-PSTM-STK-CG-TO-DIYAN.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/labeled/qyb/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:3 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/chahetai/chjSmall_mig.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/chahetai/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:4 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/gyx/GYX-small_converted.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/gyx/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:5 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/mig1100_1700/mig1100_1700.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/mig1100_1700/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:6 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/moxi/Gst_lilei-small.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/moxi/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:7 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/n2n3_small/n2n3.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/n2n3_small/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:0 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/PXZL/PXZL.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/PXZL/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:1 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/QK/RDC-premig.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/QK/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:2 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/sc/mig-small.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/sc/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:3 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/sudan/Fara_El_Harr.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/sudan/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:4 &

python ./projects/Fault_recong/predict.py --config $CONFIG \
                                        --checkpoint $CKPTS \
                                        --input /gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/unlabeled/yc/seis.sgy \
                                        --save_path $SAVE_ROOT_PATH/project_data_v1_pred/unlabeled/yc/\
                                        --predict_type 3d \
                                        --force_3_chan True \
                                        --device cuda:5 &

