import numpy as np
import h5py
import cigvis

f = h5py.File('Fault_data/public_data/crop_256/val/4.h5','r')
seis = f['raw'][:]
gt = f['label'][:]
f.close()

f = h5py.File('MIM-Med3D/output/Fault_Finetuning/swin_unetr_base_simmim500e_p16_public_256_flip_rotate_aug_4x4_rerun/thebe_chuck_pred/4.h5','r')
pred_score = f['score'][:]
f.close()

nodes, cbar = cigvis.create_slices(seis,
                                   cmap='Petrel',
                                   return_cbar=True,
                                   label_str='Amplitude')
nodes.append(cbar)

cigvis.plot3D(nodes, size=(1000, 800), savename='example.png')