import numpy as np
import segyio
import os
import argparse

def ConvertScoreToSGY(score_path, th, save_path):
    '''
    args:
        - score_path: .npy score file
        - th: threshold
        - save_path: path to save converted file
    
    '''
    print(f'Loading {score_path}')
    score = np.load(score_path)
    print('Start Convert')
    prediction = (score > th).astype(np.float32)
    save_file_path = os.path.join(save_path, f'prediction.bin')
    prediction.tofile(save_file_path)
    print(f'Saved converted file to {save_file_path}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--score_path', type=str, help='Input score prediction')
    args.add_argument('--th', type=float, help='Threshold')
    args.add_argument('--save_path', type=str, help='Path to save predictions')
    args = args.parse_args()
    
    ConvertScoreToSGY(args.score_path, args.th, args.save_path)