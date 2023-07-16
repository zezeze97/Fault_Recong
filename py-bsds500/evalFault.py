import numpy as np
import os, argparse
import tqdm
from bsds import evaluate_boundaries
import cv2
import segyio



def load_gt_boundaries(sample_name):
    gt = fault[sample_name, :, :]
    gt = gt.astype(np.float32)
    gt = cv2.resize(gt,(w_s,h_s))
    gt = gt>0.5
    gt = gt.astype(np.float32)
    return gt

def load_pred(sample_name):
    pred = prediction[sample_name, :, :]
    pred = cv2.resize(pred,(w_s,h_s))
    return pred


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gt_path', type=str)
    args.add_argument('--pred_path', type=str)
    args.add_argument('--step', type=int)
    args.add_argument('--scalefactor', type=int)
    args.add_argument('--UPPER_BOUND', type=int, default=None)
    args.add_argument('--LOWER_BOUND', type=int, default=None)
    args.add_argument('--start_idx', type=int, default=0)
    args = args.parse_args()
    
    
    N_THRESHOLDS = 99
    # Loading Fault Ground Truth
    if '.npy' in args.gt_path:
        fault = np.load(args.gt_path, mmap_mode='r')[args.start_idx:, :, :]
    elif '.sgy' in args.gt_path:
        fault = segyio.tools.cube(args.gt_path)[args.start_idx:, :, :]
    if args.UPPER_BOUND is not None and args.LOWER_BOUND is not None:
        fault = fault[:, :, args.LOWER_BOUND: args.UPPER_BOUND]
    
    
    # Loading Predictions 
    prediction = np.load(args.pred_path, mmap_mode='r')[args.start_idx:, :, :]
    if args.UPPER_BOUND is not None and args.LOWER_BOUND is not None:
        prediction = prediction[:, :, args.LOWER_BOUND: args.UPPER_BOUND]
    SAMPLE_NAMES = [i for i in range(0, fault.shape[0], args.step)]
    

    _, h,w = fault.shape

    h_s, w_s = int(h/args.scalefactor), int(w/args.scalefactor)
    print("h_s, w_s", h_s, w_s)

    sample_results, threshold_results, overall_result = evaluate_boundaries.pr_evaluation(N_THRESHOLDS, SAMPLE_NAMES,
                                                                                        load_gt_boundaries, load_pred,
                                                                                        progress=tqdm.tqdm, apply_thinning=False)

    overallresults = np.zeros((N_THRESHOLDS,4))



    print('Per image:')
    print("res.threshold, res.recall, res.precision, res.f1")
    for sample_index, res in enumerate(sample_results):
        print('{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            sample_index + 1, res.threshold, res.recall, res.precision, res.f1))

    i=0
    print('')
    print('Overall:')
    print("res.threshold, res.recall, res.precision, res.f1")
    for thresh_i, res in enumerate(threshold_results):
        overallresults[i] = [res.threshold, res.recall, res.precision, res.f1]
        i+=1
        print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            res.threshold, res.recall, res.precision, res.f1))

    print('')
    print('Summary:')
    print(args.pred_path)
    print("threshold, recall, precision, f1, best_recall, best_precision, best_f1, area_pr")
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'
        '{:<10.6f}'.format(
        overall_result.threshold, overall_result.recall,
        overall_result.precision, overall_result.f1, overall_result.best_recall,
        overall_result.best_precision, overall_result.best_f1,
        overall_result.area_pr)
    )
