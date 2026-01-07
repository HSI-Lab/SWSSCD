# utils/metrics.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# def compute_cd_metrics(pred_mask, GT):
#     """
#     pred_mask: (H,W) or (N,) 0/1
#     GT: torch.Tensor (H,W) 0/1
#     """
#     GT = GT.cpu().numpy().flatten()
#     pred = pred_mask.flatten()

#     OA = accuracy_score(GT, pred)
#     F1 = f1_score(GT, pred)
#     P  = precision_score(GT, pred)
#     R  = recall_score(GT, pred)

#     cm = confusion_matrix(GT, pred)
#     if cm.size == 4:
#         tn, fp, fn, tp = cm.ravel()
#         IoU_change = tp / (tp + fp + fn + 1e-6)
#     else:
#         IoU_change = 0.0

#     return {
#         "OA": OA,
#         "F1": F1,
#         "Precision": P,
#         "Recall": R,
#         "IoU_change": IoU_change
#     }

def compute_cd_metrics(pred, GT):
    # ---- Convert to CPU numpy ----
    if isinstance(GT, torch.Tensor):
        GT = GT.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    GT = GT.flatten()
    pred = pred.flatten()

    OA = accuracy_score(GT, pred)
    F1 = f1_score(GT, pred)
    P  = precision_score(GT, pred)
    R  = recall_score(GT, pred)

    # IoU
    cm = confusion_matrix(GT, pred)
    tn, fp, fn, tp = cm.ravel()
    IoU_change = tp / (tp + fp + fn + 1e-6)

    return {
        "OA": OA,
        "F1": F1,
        "Precision": P,
        "Recall": R,
        "IoU_change": IoU_change
    }

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_cd_metrics(pred, GT):
    """
    支持两种 GT 编码：
    1) GT ∈ {0,1,2}: 0 忽略, 1=unchanged, 2=changed
       -> 评估时忽略 GT==0，并将 GT: 1->0, 2->1
    2) GT ∈ {0,1}: 标准二分类，直接评估

    pred 默认应为 {0,1}（若 pred 是概率图，请外部先 threshold）
    """

    # ---- Convert to CPU numpy ----
    if isinstance(GT, torch.Tensor):
        GT = GT.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    GT = np.asarray(GT).flatten()
    pred = np.asarray(pred).flatten()

    # ---- Detect GT encoding & prepare eval mask ----
    gt_unique = set(np.unique(GT).tolist())

    if gt_unique.issubset({0, 1, 2}) and (2 in gt_unique):
        # Case: 0/1/2 with ignore=0
        valid_mask = (GT != 0)
        GT_eval = GT[valid_mask]
        pred_eval = pred[valid_mask]

        # 显式逐类映射（你要求的风格）：1->1, 2->0
        GT_bin = np.zeros_like(GT_eval, dtype=np.int32)
        GT_bin[GT_eval == 1] = 1
        GT_bin[GT_eval == 2] = 0
        GT_eval = GT_bin
    else:
        # Case: 0/1
        GT_eval = GT.astype(np.int32)
        pred_eval = pred.astype(np.int32)

    if GT_eval.size == 0:
        raise ValueError("No valid pixels for evaluation (GT may be all zeros).")

    # ---- Ensure binary pred (robustness) ----
    pred_eval = (pred_eval > 0).astype(np.int32)
    GT_eval = (GT_eval > 0).astype(np.int32)

    # ---- Metrics ----
    OA = accuracy_score(GT_eval, pred_eval)
    F1 = f1_score(GT_eval, pred_eval, zero_division=0)
    P  = precision_score(GT_eval, pred_eval, zero_division=0)
    R  = recall_score(GT_eval, pred_eval, zero_division=0)

    # IoU (change class = 1)
    cm = confusion_matrix(GT_eval, pred_eval, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    IoU_change = tp / (tp + fp + fn + 1e-6)

    return {
        "OA": OA,
        "F1": F1,
        "Precision": P,
        "Recall": R,
        "IoU_change": IoU_change
    }
