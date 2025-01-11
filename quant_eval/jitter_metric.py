import numpy as np


def calculate_jitter(pred_motion, gt_motion):
    "motion: seq_len, joints, 3"
    l1_diff_pred = np.abs(pred_motion[1:] - pred_motion[:-1])
    l1_diff_gt = np.abs(gt_motion[1:] - gt_motion[:-1])

    return np.mean(np.abs(l1_diff_pred - l1_diff_gt))
