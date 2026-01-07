import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from skimage.filters import threshold_otsu

from datasets.farmland_cd_dataset import HSICD_SelfSupDataset
from models.simple_encoder import SimpleConvEncoder
from utils.metrics import compute_cd_metrics
from utils.visualization import save_prediction_images

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False


def compute_encoder_gflops_and_params(model: torch.nn.Module, x: torch.Tensor):
    """
    Compute encoder-only GFLOPs per single forward pass, plus parameter count.
    Notes:
      - Uses fvcore's FlopCountAnalysis (counts FLOPs for the model forward).
      - Returns:
          gflops_per_forward: float or None
          params_m: float or None
    """
    if not _HAS_FVCORE:
        return None, None

    try:
        flops = FlopCountAnalysis(model, x).total()  # FLOPs (not MACs)
        gflops_per_forward = float(flops) / 1e9
        params_dict = parameter_count(model)
        total_params = 0
        for _, v in params_dict.items():
            total_params += int(v)
        params_m = float(total_params) / 1e6

        return gflops_per_forward, params_m
    except Exception as e:
        print(f"[WARN] GFLOPs/Params computation failed: {e}")
        return None, None


def test_selfsup_cd(
    mat_path,
    mask_path,
    ckpt_path,
    device,
    out_root
):

    dataset = HSICD_SelfSupDataset(mat_path, mask_path, pca_dim=128, pca_cache="./pca_farmland.npy")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=("cuda" in device))

    C = dataset.B  # HSI 光谱通道数
    model = SimpleConvEncoder(in_channels=C).to(device)

    print(f"[Test] Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    gflops_per_forward = None
    params_m = None
    if _HAS_FVCORE:
        try:
            first_batch = next(iter(loader))
            t1_probe = first_batch["t1"].to(device)  # [1, C, H, W]
            gflops_per_forward, params_m = compute_encoder_gflops_and_params(model, t1_probe)

            if gflops_per_forward is not None:
                print(f"[Test] Encoder GFLOPs (per forward): {gflops_per_forward:.6f}")
            if params_m is not None:
                print(f"[Test] Encoder Params (M): {params_m:.6f}")
        except StopIteration:
            print("[WARN] Empty dataloader; skip GFLOPs computation.")
    else:
        print("[WARN] fvcore not installed, skip GFLOPs/Params computation. Install via: pip install fvcore")

    results_dir = os.path.join(out_root, "test_results")
    logs_dir = os.path.join(out_root, "test_logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "test_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_f:
        if gflops_per_forward is not None:
            log_f.write(f"[Encoder] GFLOPs_per_forward={gflops_per_forward:.6f}\n")
        if params_m is not None:
            log_f.write(f"[Encoder] Params_M={params_m:.6f}\n")

        print("[Test] Start Testing on dataset A ...")

        all_metrics = []

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                t1 = batch["t1"].to(device)          # [1, C, H, W]
                t2 = batch["t2"].to(device)          # [1, C, H, W]
                GT = batch["GT"].to(device)          # [H, W] or [1, H, W]

                s = time.time()
                f1 = model(t1)                       # [1, D, H, W]
                f2 = model(t2)                       # [1, D, H, W]
                infer_time = time.time() - s

                score_map = torch.norm(f1 - f2, dim=1).squeeze().cpu().numpy()

                thresh = threshold_otsu(score_map)
                pred_mask = (score_map > thresh).astype(np.uint8)

                GT_np = GT.squeeze().cpu().numpy().astype(np.uint8)
                metrics = compute_cd_metrics(pred_mask, GT_np)
                all_metrics.append(metrics)

                save_prediction_images(
                    epoch=f"test_{idx:03d}",
                    score_map=score_map,
                    pred_mask=pred_mask,
                    out_dir=results_dir
                )

                log_line = (
                    f"[Sample {idx:03d}] "
                    f"OA={metrics['OA']:.6f} "
                    f"F1={metrics['F1']:.6f} "
                    f"Precision={metrics['Precision']:.6f} "
                    f"Recall={metrics['Recall']:.6f} "
                    f"IoU={metrics['IoU_change']:.6f} "
                    f"Th={thresh:.6f} "
                    f"Time={infer_time:.6f}s\n"
                )
                print(log_line.strip())
                log_f.write(log_line)

        if len(all_metrics) > 0:
            mean_OA = np.mean([m["OA"] for m in all_metrics])
            mean_F1 = np.mean([m["F1"] for m in all_metrics])
            mean_P = np.mean([m["Precision"] for m in all_metrics])
            mean_R = np.mean([m["Recall"] for m in all_metrics])
            mean_IoU = np.mean([m["IoU_change"] for m in all_metrics])

            summary_line = (
                f"[Test Summary] "
                f"OA={mean_OA:.6f} "
                f"F1={mean_F1:.6f} "
                f"Precision={mean_P:.6f} "
                f"Recall={mean_R:.6f} "
                f"IoU={mean_IoU:.6f}\n"
            )
            print(summary_line.strip())
            log_f.write(summary_line)

    print(f"[Test] Done. Logs saved to {log_path}, results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mat_path", type=str, default="./data/Farmland.mat")
    parser.add_argument("--out_root", type=str, default="./outputs")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/Farmland.pth"
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_path = "./masks/Farmland_mask.png"

    test_selfsup_cd(
        mat_path=args.mat_path,
        mask_path=mask_path,
        ckpt_path=args.ckpt_path,
        device=device,
        out_root=args.out_root
    )
