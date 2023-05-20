GPU=$2
port=23489


config=simmim_swin-base-w6_100e_512x512_public

if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PORT=${port} ./tools/dist_train.sh ./projects/Fault_Recong/config/${config}.py 1 --work-dir output/${config} 
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128.py ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/iter_48000.pth 1
fi