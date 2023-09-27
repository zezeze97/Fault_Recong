import h5py
import cv2
import os
from tqdm import tqdm
import numpy as np
import segyio

def main_v0():
    scr_root_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/crop'
    dst_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/crop_2d_slices'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))

    
    # convert train data
    data_lst = os.listdir(os.path.join(scr_root_path, 'train'))
    for item in tqdm(data_lst):
        with h5py.File(os.path.join(scr_root_path, 'train', item), 'r') as f:
            image_cube = f['raw'][:]
            label = f['label'][:]
            # label = label.squeeze(0)
        num_id = int(item.split('.')[0])
        for i in range(128):
            label_slice = label[:,:,i]
            if np.sum(label_slice) > 0.03 * 128 * 128:
                image_slice = image_cube[:,:,i]
                # [0-1] scale
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = image_slice * 255
                
                cv2.imwrite(os.path.join(dst_path, 'train', 'image', f'cube_{num_id}_slice_{i}.png'), image_slice)
                cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'cube_{num_id}_slice_{i}.png'), label_slice)
            
            
    
    # convert val data
    data_lst = os.listdir(os.path.join(scr_root_path, 'val'))
    for item in tqdm(data_lst):
        with h5py.File(os.path.join(scr_root_path, 'val', item), 'r') as f:
            image_cube = f['raw'][:]
            label = f['label'][:]
            # label = label.squeeze(0)
        num_id = int(item.split('.')[0])
        for i in range(128):
            image_slice = image_cube[:,:,i]
            # [0-1] scale
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            image_slice = image_slice * 255
            label_slice = label[:,:,i]
            cv2.imwrite(os.path.join(dst_path, 'val', 'image', f'cube_{num_id}_slice_{i}.png'), image_slice)
            cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'cube_{num_id}_slice_{i}.png'), label_slice)
    
def main_v1():
    scr_root_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed'
    dst_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/2d_slices'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))
    print('loading seis train data')
    seis_train = np.load(os.path.join(scr_root_path, 'seistrain.npy'))
    fault_train = np.load(os.path.join(scr_root_path, 'faulttrain.npy'))
    assert seis_train.shape == fault_train.shape
    for i in tqdm(range(seis_train.shape[0])):
        seis_slice = seis_train[i,:,:]
        # seis_slice = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min())
        fault_slice = fault_train[i,:,:]
        # convert to gray
        # seis_slice = 255 * seis_slice
        np.save(os.path.join(dst_path, 'train', 'image', f'{i}.npy'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'{i}.png'), fault_slice)
    del seis_train
    del fault_train
    
    print('loading seis val data')
    seis_val = np.load(os.path.join(scr_root_path,'seisval.npy'))
    fault_val = np.load(os.path.join(scr_root_path, 'faultval.npy'))
    assert seis_val.shape == fault_val.shape
    # seis_val = (seis_val - seis_val.min()) / (seis_val.max() - seis_val.min())
    for i in tqdm(range(seis_val.shape[0])):
        seis_slice = seis_val[i,:,:]
        # seis_slice = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min())
        fault_slice = fault_val[i,:,:]
        # convert to gray
        # seis_slice = 255 * seis_slice
        np.save(os.path.join(dst_path, 'val', 'image', f'{i}.npy'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'{i}.png'), fault_slice)
    del seis_val
    del fault_val

def main_v2():
    data_root = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data'
    dst_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/2d_slices'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))
    seis_data = segyio.tools.cube(os.path.join(data_root, 'origin_data', 'seis', 'mig_fill.sgy'))
    # precess missing value
    # seis_data[seis_data==-912300] = seis_data[seis_data!=-912300].mean()
    # seis_data[seis_data==0.0] = seis_data[seis_data!=0.0].mean()
    fault = segyio.tools.cube(os.path.join(data_root, 'origin_data', 'fault','label_fill.sgy'))
    fault = fault.astype(np.uint8)
    for i in range(373):
        seis_slice = seis_data[i,:,:]
        fault_slice = fault[i,:,:]
        np.save(os.path.join(dst_path, 'train', 'image', f'{i}.npy'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'{i}.png'), fault_slice)
    k = 0
    for i in range(373,501):
        seis_slice = seis_data[i,:,:]
        fault_slice = fault[i,:,:]
        # convert to gray
        np.save(os.path.join(dst_path, 'val', 'image', f'{k}.npy'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'{k}.png'), fault_slice)
        k += 1

def convert_2d_ssl(root_dir, seis_name):
    dst_path = os.path.join(root_dir, '2d_slices_ssl')
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
    # seis_data = segyio.tools.cube(os.path.join(root_dir, seis_name))
    seis_data = segyio.tools.cube(os.path.join(root_dir, 'seis', seis_name))
    print(f'Input seis shape is {seis_data.shape}')
    iline, xline, timelne = seis_data.shape
    for i in tqdm(range(iline)):
        seis_slice = seis_data[i, :, :]
        np.save(os.path.join(dst_path, 'train', 'image', f'{i}.npy'), seis_slice)

def convert_2d_sl(root_dir, seis_name, fault_name, start_id, end_id, step, convert_fault=False):
    dst_path = os.path.join(root_dir, '2d_slices_sl')
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
    # seis = segyio.tools.cube(os.path.join(root_dir, 'seis', seis_name))
    # fault = segyio.tools.cube(os.path.join(root_dir, 'faults', fault_name))
    seis = segyio.tools.cube(os.path.join(root_dir, seis_name))
    fault = segyio.tools.cube(os.path.join(root_dir, fault_name))
    assert seis.shape == fault.shape
    print(f'shape is {seis.shape}')
    for i in tqdm(range(start_id, end_id, step)):
        seis_slice = seis[i, :, :]
        fault_slice = fault[i, :, :]
        if convert_fault:
            fault_slice = fault_pre_process(fault_slice)
        assert np.sum(fault_slice) > 0.0
        np.save(os.path.join(dst_path, 'train', 'image', f'{i}.npy'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'{i}.png'), fault_slice)

def fault_pre_process(fault):
    '''
    input: fault, 2d fault
    output: processed 2d fault
    '''
    shape = fault.shape
    processed_fault = np.zeros(shape, dtype=np.uint8)
    processed_fault[fault==2.0] = 1
    processed_fault = processed_fault.astype(np.uint8)
    return processed_fault

if __name__ == '__main__':
    convert_2d_sl(root_dir='/home/zhangzr/FaultRecongnition/Fault_data/ODData/dafeng1/Export',
                  seis_name='seis.sgy',
                  fault_name='FaultVolume.sgy',
                  start_id=0,
                  end_id=1320,
                  step=40,
                  convert_fault=True)




    '''
    convert_2d_sl(root_dir='./Fault_data/project_data_v2/GYX',
                  seis_name='GYX3D2018-PSDM-VTI-CG1203-400Km2-DP-50.sgy',
                  fault_name='faults.sgy',
                  start_id=7,
                  end_id=916,
                  step=8)
    convert_2d_sl(root_dir='./Fault_data/project_data_v2/LH3D',
                  seis_name='TJ-2022-6-15-pstm-cg.sgy',
                  fault_name='faults.sgy',
                  start_id=4,
                  end_id=701,
                  step=8)
    convert_2d_sl(root_dir='./Fault_data/project_data_v2/ZG3D',
                  seis_name='yanfa__ZG3d_PSTM_CG_0715-small.sgy',
                  fault_name='faults.sgy',
                  start_id=4,
                  end_id=1101,
                  step=8)

    unlabeled_root_dir = '/gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v2'
    dir_name_lst = ['dafeng1', 'gaojiapu', 'guai3east', 'GYX', 'LH3D', 'madonglianpian', 'moxi', 'pingliang', 'ZG3D']
    seis_name_lst = ['C06_DF1J_OVT_PSTM_GAIN_TJ20220704_-1.sgy', 'gjp_seis.sgy', 'PSTM_gain_20141128_shift1080ms_16f_111.sgy', 'GYX3D2018-PSDM-VTI-CG1203-400Km2-DP-50.sgy', 'TJ-2022-6-15-pstm-cg.sgy', 'PSTM_gain_20230324_shift1000ms_1.sgy', 'moxi_seisl_converted.sgy', 'pingliang_seis_converted.sgy', 'yanfa__ZG3d_PSTM_CG_0715-small.sgy']
    for i, dir_name in enumerate(dir_name_lst): 
        root_dir = os.path.join(unlabeled_root_dir, dir_name)
        print(f'Processing {root_dir}')
        convert_2d_ssl(root_dir, seis_name_lst[i])
    
    # root_dir = '/gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/labeled/qyb'
    # seis_name = '20230412_QY-PSTM-STK-CG-TO-DIYAN.sgy'
    # convert_2d_ssl(root_dir, seis_name)
    
    root_dir = '/gpfs/share/home/2001110054/Fault_Recong/Fault_data/project_data_v1/labeled/Ordos/yw'
    seis_name = 'mig.sgy'
    fault_name = 'fault_volume_converted.sgy'
    start_id = 0
    end_id = 673
    step = 8
    convert_2d_sl(root_dir, seis_name, fault_name, start_id, end_id, step)
    '''