# utils/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def save_prediction_images(epoch, score_map, pred_mask, out_dir, dpi=300):
    """
    score_map: (H,W) float
    pred_mask: (H,W) 0/1
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.imshow(score_map, cmap="hot")
    plt.colorbar()
    plt.title(f"Epoch {epoch} Score Map")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"epoch_{epoch}_score.png"), 
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.close()


    pred_img = (pred_mask * 255).astype(np.uint8)
    img = Image.fromarray(pred_img)

    img.save(
        os.path.join(out_dir, f"epoch_{epoch}_pred.png"),
        dpi=(dpi, dpi)
    )
