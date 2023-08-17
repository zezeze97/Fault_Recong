import numpy as np
import segyio
import argparse

def ConvertScoreToSGY(score_path, th, save_path):
    '''
    args:
        - score_path: .npy score file
        - th: threshold
        - save_path: path to save converted file
    
    '''
    score = np.load(score_path)
    prediction = (score > th).astype(np.uint8)
    # segyio.open(save_path, )