import numpy as np
import os
from tqdm import tqdm

def main(src_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    fault = np.load(src_path, mmap_mode='r')
    print(fault.shape)
    k = 0
    for i in tqdm(range(0, fault.shape[0], 5)):
        np.save(os.path.join(target_path, f'{k}.npy'), fault[i, :, :])
        k += 1

if __name__ == '__main__':
    src_path = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_force_3_chan-512x512_per_image_normal_pos_weight_10/thebe_pred/score.npy'
    target_path = '/home/zhangzr/Fault_Recong/py-bsds500/swin_pred'
    main(src_path, target_path)
        