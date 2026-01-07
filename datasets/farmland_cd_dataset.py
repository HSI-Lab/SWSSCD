import os
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import joblib


class HSICD_SelfSupDataset(Dataset):
    def __init__(self, mat_path, mask_path, pca_dim=32, pca_cache="./pca_farmland.pkl"):
        super().__init__()

        mat = sio.loadmat(mat_path)

        data_T1 = mat['T1']  # (H, W, B)
        data_T2 = mat['T2']
        GT = mat['GT']       # (H, W), 0/1

        self.GT = torch.from_numpy(GT.astype(np.uint8))

        assert data_T1.shape == data_T2.shape
        H, W, B = data_T1.shape

        eps = 1e-6
        minv = min(data_T1.min(), data_T2.min())
        maxv = max(data_T1.max(), data_T2.max())
        
        data_T1 = (data_T1 - minv) / (maxv - minv + eps)
        data_T2 = (data_T2 - minv) / (maxv - minv + eps)

        if pca_dim is not None:
            if not os.path.exists(pca_cache):
                print(f"[Dataset] Fitting PCA to {pca_dim} dimensions ...")

                X = data_T1.reshape(-1, B)
                pca = PCA(n_components=pca_dim)
                pca.fit(X)

                joblib.dump(pca, pca_cache)
                print(f"[Dataset] PCA object saved to {pca_cache}")
            else:
                print(f"[Dataset] Loading PCA object from {pca_cache}")
                pca = joblib.load(pca_cache)

            t1_pca = pca.transform(data_T1.reshape(-1, B)).reshape(H, W, pca_dim)
            t2_pca = pca.transform(data_T2.reshape(-1, B)).reshape(H, W, pca_dim)

            self.t1 = torch.from_numpy(t1_pca).permute(2, 0, 1).float()  # [C, H, W]
            self.t2 = torch.from_numpy(t2_pca).permute(2, 0, 1).float()
            self.B = pca_dim
        else:
            self.t1 = torch.from_numpy(data_T1).permute(2, 0, 1).float()
            self.t2 = torch.from_numpy(data_T2).permute(2, 0, 1).float()
            self.B = B

        mask_img = Image.open(mask_path).convert('L')
        mask = np.array(mask_img)

        self.stable_mask = torch.from_numpy(mask == 0)
        self.unstable_mask = torch.from_numpy(mask == 1)

        self.H, self.W = H, W

    def __len__(self):
        return 1  # whole image

    def __getitem__(self, idx):
        return {
            "t1": self.t1,
            "t2": self.t2,
            "stable_mask": self.stable_mask,
            "unstable_mask": self.unstable_mask,
            "GT": self.GT
        }



class HSICD_SelfSupDataset1(Dataset):
    def __init__(self, mat_path, mask_path):
        super().__init__()
        mat = sio.loadmat(mat_path)

        data_T1 = mat['T1']       # (H,W,B)
        data_T2 = mat['T2']
        GT = mat['GT']

        self.GT = torch.from_numpy(GT.astype(np.uint8))

        assert data_T1.shape == data_T2.shape
        H, W, B = data_T1.shape

        minv = min(data_T1.min(), data_T2.min())
        maxv = max(data_T1.max(), data_T2.max())
        eps = 1e-6

        data_T1 = (data_T1 - minv) / (maxv - minv + eps)
        data_T2 = (data_T2 - minv) / (maxv - minv + eps)

        self.t1 = torch.from_numpy(data_T1).permute(2,0,1).float()
        self.t2 = torch.from_numpy(data_T2).permute(2,0,1).float()

        mask_img = Image.open(mask_path).convert('L')
        mask = np.array(mask_img)

        self.stable_mask = torch.from_numpy(mask == 0)
        self.unstable_mask = torch.from_numpy(mask == 1)

        self.H, self.W, self.B = H, W, B

    def __len__(self):
        return 1  # whole image

    def __getitem__(self, idx):
        return {
            "t1": self.t1,
            "t2": self.t2,
            "stable_mask": self.stable_mask,
            "unstable_mask": self.unstable_mask,
            "GT": self.GT
        }
