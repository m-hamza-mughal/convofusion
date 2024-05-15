import numpy as np
import glob
import os
from tqdm import tqdm

def calculate_jitter(pred_motion, gt_motion):
    "motion: seq_len, joints, 3"
    sq_diff_pred = np.abs(pred_motion[1:] - pred_motion[:-1]) 
    sq_diff_gt = np.abs(gt_motion[1:] - gt_motion[:-1])
    # sq_diff_pred = (pred_motion[1:] - pred_motion[:-1])**2
    # sq_diff_gt = (gt_motion[1:] - gt_motion[:-1])**2

    return np.mean(np.abs(sq_diff_pred - sq_diff_gt))

