from multi_seg_main import MultiSegtrainer
from seg_multi_decoder_main import MultiDecoderSegtrainer
from seg_multi_decoder_fusion_main import MultiDecoderFusionSegtrainer
from seg_cls_main import SegCls_trainer
import yaml
from pytorch_lightning import Trainer
from data.Fault_dataset import FaultDataset
import os
import numpy as np
import h5py
import shutil
import torch
from tqdm import tqdm
from monai.transforms import NormalizeIntensity, Zoom, Compose
import segyio


def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs):
        """
        :param raw_dataset: ndarray of raw data
        :param label_dataset: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset, patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'

def predict(config_path, ckpt_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(config_path,encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    model = MultiSegtrainer(**config['model']['init_args'])
    trainer = Trainer(accelerator='gpu', devices=-1, default_root_dir=output_path, inference_mode=True)
    # trainer = Trainer(accelerator='cpu', devices=None, default_root_dir=output_path, inference_mode=True)
    
    dataloader = FaultDataset(**config['data']['init_args'])
    dataloader.setup(stage='test')
    pred_loader = dataloader.test_dataloader()
    preds = trainer.predict(model, pred_loader,ckpt_path=ckpt_path)
    for item in preds:
        for image_name in item.keys():
            pred = item[image_name]['pred']
            score = item[image_name]['score']
            pred = pred.squeeze(0) # (128, 128, 128)
            score = score.squeeze(0) # (128, 128, 128)
            with h5py.File(os.path.join(output_path, image_name.split('.')[0]+'.h5'), 'w') as f:
                f['predictions'] = pred.cpu().numpy()
                f['score'] = score.cpu().numpy()
    shutil.rmtree(os.path.join(output_path, 'lightning_logs'))
    # print(preds)

def predict_sliding_window(config_path, ckpt_path, input_path, output_path, device='cuda:0'):
    
    # load seis
    if isinstance(input_path, str):
        print(f'Loading image from {input_path}')
        if '.npy' in input_path:
            seis = np.load(input_path, mmap_mode='r') # [:, :, 400:1500]
        elif '.sgy' in input_path:
            # seis = segyio.tools.cube(input_path)[373:,:,:]
            seis = segyio.tools.cube(input_path)
        else:
            raise TypeError
    else:
        seis = input_path
    
    # load config 
    with open(config_path,encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    # get slice builder 
    slice_builder = SliceBuilder(raw_dataset=seis,
                                label_dataset=None,
                                weight_dataset=None,
                                patch_shape=(128, 128, 128),
                                stride_shape=(64, 64, 64)
                                # patch_shape=(256, 256, 256),
                                # stride_shape=(128, 128, 128)
                                )
    crop_cubes_pos = slice_builder.raw_slices
    
    # create missing dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # init model
    model_name = config['model']['init_args']['model_name']
    if model_name == 'swin_unetr_multi_decoder':
        model = MultiDecoderSegtrainer(**config['model']['init_args'])
    elif model_name == 'swin_unetr_seg_cls':
        model = SegCls_trainer(**config['model']['init_args'])
    elif model_name == 'swin_unetr_multi_decoder_fusion':
        model = MultiDecoderFusionSegtrainer(**config['model']['init_args'])
    else:
        model = MultiSegtrainer(**config['model']['init_args'])
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    model.to(device)
    model.eval()
    
    # data preprocess
    # mean=1.7283046245574951
    # std=6800.84033203125
    # data_preprocess = NormalizeIntensity(subtrahend=mean, divisor=std, nonzero=False, channel_wise=False)
    data_preprocess = NormalizeIntensity(nonzero=True, channel_wise=False)
    
    # predict
    output_logits = np.zeros(seis.shape)
    count_mat = np.zeros(seis.shape, dtype=np.uint8)
    with torch.no_grad():
        for pos in tqdm(crop_cubes_pos):
            x_range = pos[0]
            y_range = pos[1]
            z_range = pos[2]
            seis_cube_crop = seis[x_range, y_range, z_range]
            # preprocess 
            seis_cube_crop = torch.from_numpy(seis_cube_crop.copy()).unsqueeze(0) # [C, H, W, D]
            input = data_preprocess(seis_cube_crop)  
            input = input.unsqueeze(0) # batch size = 1
            # forward
            if model_name in ['swin_unetr_multi_decoder', 'swin_unetr_seg_cls', 'swin_unetr_multi_decoder_fusion']:
                logits = model.forward_test(input.to(device))
            else:
                logits = model(input.to(device))
            # post process
            logits = logits.squeeze(0) # move batch dim
            logits = logits.squeeze(0) # move channel
            logits = logits.detach().cpu().numpy()
            # fill into origin pred
            output_logits[x_range, y_range, z_range] += logits
            count_mat[x_range, y_range, z_range] += 1
    output_logits /= count_mat
    # output_pred = model.post_trans(output_logits).cpu().numpy()
    output_score = model.post_score_trans(output_logits).cpu().numpy()
    # convert to float 16
    output_score = output_score
    # np.save(os.path.join(output_path, input_path.split('/')[-1].split('.')[0]+'_pred.npy'), output_pred)
    np.save(os.path.join(output_path, input_path.split('/')[-1].split('.')[0]+'_score.npy'), output_score)
    
   #  with h5py.File(os.path.join(output_path, input_path.split('/')[-1].split('.')[0]+'.h5'), 'w') as f:
        # f['predictions'] = output_pred
        # f['score'] = output_score
    
            
        
            
            
    
        


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description='Using 3D segmentation to predict 3D Fault')
    args.add_argument('--config', type=str, help='model config file path')
    args.add_argument('--checkpoint', type=str, help='model checkpoint path')
    args.add_argument('--input', type=str, help='input image/cube path')
    args.add_argument('--save_path', type=str, help='path to save predict result')
    args.add_argument('--device', default='cuda:0')
    args = args.parse_args()
    
    
    config_path = args.config
    ckpt_path = args.checkpoint
    input_path = args.input
    output_path = args.save_path
    device = args.device
    # gt_path = '/home/zhangzr/Fault_Recong/Fault_data/public_data/precessed/test/fault/faulttest.npy'
    # predict(config_path, ckpt_path, output_path)
    predict_sliding_window(config_path, ckpt_path, input_path, output_path, device)