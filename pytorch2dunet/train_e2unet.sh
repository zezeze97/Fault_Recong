python train.py --model_type e2UNet \
                --train_dir /home/zhangzr/FaultRecongnition/Fault_data/2d-simulate-data/train \
                --val_dir /home/zhangzr/FaultRecongnition/Fault_data/2d-simulate-data/val \
                --ckpt_save_dir e2UNet_CKPTS \
                --batch-size 4 --amp --classes 1 --epochs 20 --bilinear