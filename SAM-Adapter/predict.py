import argparse
import torch
import torch.nn.functional as F
import os
import cv2
from tqdm import tqdm
import numpy as np
import segyio
from models.sam import sam_model_registry


        
def random_click(mask, point_labels = 1, inout = 1):
    indices = np.argwhere(mask == inout)
    return indices[np.random.randint(len(indices))]


def slice_inference(slice, net, device, patch_size=1024, stride=512):
    h_stride, w_stride = stride, stride
    h_crop, w_crop = patch_size, patch_size
    batch_size, _, h_img, w_img = slice.size()
    num_classes = 1
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = slice.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = slice.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = slice[:, :, y1:y2, x1:x2]
            crop_seg_logit = inference(crop_img, net, device)
            preds += F.pad(crop_seg_logit,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    seg_logits = preds / count_mat
    output = torch.sigmoid(seg_logits).cpu().squeeze(0).squeeze(0).numpy()
    return output


def inference(img, net, device):
    b, c, h, w = img.shape
    pt = random_click(np.ones((h, w)), point_labels=1, inout=1)
    pt = torch.from_numpy(pt).unsqueeze(0).to(device)
    point_label = torch.tensor([1]).to(device)
    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
    labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=device)
    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    pt = (coords_torch, labels_torch)
    
    imge= net.image_encoder(img)

    se, de = net.prompt_encoder(
        points=pt,
        boxes=None,
        masks=None,
    )

    pred, _ = net.mask_decoder(
        image_embeddings=imge,
        image_pe=net.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de, 
        multimask_output=False,
    )
    pred = F.interpolate(pred, scale_factor=4, mode='nearest')
    return pred
    
    
    




def main(args):
    
    
    # create save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # set up model
    
    args.image_size = 1024
    args.out_size = 256
    args.patch_size = 2
    args.dim = 512
    args.depth = 1
    args.heads = 16
    args.mlp_dim = 1024
    args.thd = False
    net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt).to(args.device)
    net.eval()
    
    # load input 
    if '.sgy' in args.input_cube_path:
        seis = segyio.tools.cube(args.input_cube_path)
    elif '.npy' in args.input_cube_path:
        seis = np.load(args.input_cube_path, mmap_mode='r')
    
    output = []
    with torch.no_grad():
        for i in tqdm(range(seis.shape[0])):
            slice = seis[i, :, :]
            # convert to 0-255
            slice = 255 * (slice - slice.min()) / (slice.max() - slice.min())
            # stack
            slice = np.stack([slice, slice, slice], axis=0)
            slice = slice.astype(np.uint8)

            # move to tensor and scale
            slice = torch.from_numpy(slice) / 255
            slice = slice.unsqueeze(0).to(args.device)
            pred = slice_inference(slice, net, device=args.device)
            output.append(pred)
    
    output = np.stack(output, axis=0)
    np.save(os.path.join(args.save_path, 'score.npy'), output)
            
            
            
            
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device')
    parser.add_argument('--input_cube_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--sam_ckpt', type=str, default='./logs/Fault2D_SAM_2023_11_10_01_40_49/Model/best_checkpoint.pth')
    args = parser.parse_args()
    main(args)
    