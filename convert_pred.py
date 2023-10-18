import numpy as np
import segyio
import os
import argparse
import cv2
from tqdm import tqdm

def ConvertScoreToBin(score_path, th, save_path, erode=False):
    '''
    args:
        - score_path: .npy score file
        - th: threshold
        - save_path: path to save converted file
    
    '''
    print(f'Loading {score_path}')
    score = np.load(score_path)
    print('Start Convert')
    # prediction = (score > th).astype(np.float32)
    prediction = (score > th).astype(np.uint8)
    if erode:
        kernel = np.ones((3, 3), dtype=np.uint8)
        for i in tqdm(range(prediction.shape[1])):
            prediction[:, i, :] = cv2.erode(prediction[:, i, :], kernel, iterations=3)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, f'prediction.bin')
    prediction = prediction.astype(np.float32)
    prediction.tofile(save_file_path)
    print(f'Saved converted file to {save_file_path}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--score_path', type=str, help='Input score prediction')
    args.add_argument('--th', type=float, help='Threshold')
    args.add_argument('--erode', action='store_true')
    args.add_argument('--save_path', type=str, help='Path to save predictions')
    args = args.parse_args()
    
    ConvertScoreToBin(args.score_path, args.th, args.save_path, args.erode)