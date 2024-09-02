import h5py
import os
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import torch
import torch.nn.functional as F
from einops import rearrange


class NSTK(torch.utils.data.Dataset):
    def __init__(
        self,
        patch_size=256,
        stride = 128,
        scratch_dir='/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/',
        size=1000,
        window=1,
        horizon=1,
    ):
        super(NSTK, self).__init__()
        
        self.paths = [
            os.path.join(scratch_dir,'1000_2048_2048_seed_2150.h5'),
            os.path.join(scratch_dir,'8000_2048_2048_seed_2150.h5'),
            os.path.join(scratch_dir,'16000_2048_2048_seed_2150.h5'),
        ]

        self.RN = [1000,8000,32000]

        self.size = size

        self.window = window
        self.horizon = horizon
        self.trajectory_length = 16
        
        self.patch_size = patch_size
        self.stride = stride

        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape
        self.rows = self.data_shape[-2]
        self.cols = self.data_shape[-1]

        self.max_row = (self.rows - self.patch_size) // self.stride + 1
        self.max_col = (self.cols - self.patch_size) // self.stride + 1

        self.open_hdf5()
    
    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]

    """
    def extract_patches(self):
        # Assuming B is the number of datasets (RNs) and the shape of each dataset is (B, H, W)
        patches = []
        
        for dataset in self.datasets:
            B, H, W = dataset.shape
            
            # Calculate the number of patches
            patch_H, patch_W = self.patch_size, self.patch_size
            stride_H, stride_W = self.stride, self.stride
            
            out_H = (H - patch_H) // stride_H + 1
            out_W = (W - patch_W) // stride_W + 1
            P = out_H * out_W  # Total number of patches per dataset
            
            # Use sliding_window_view to extract patches
            dataset_patches = stride_tricks.sliding_window_view(
                dataset, (patch_H, patch_W), axis=(1, 2)
            )[::stride_H, ::stride_W]
            
            # Reshape patches to have shape (B * P, patch_H, patch_W)
            dataset_patches = dataset_patches.reshape(-1, patch_H, patch_W)
            print(dataset_patches.shape)
            patches.append(dataset_patches)
        
        # Concatenate all patches along the first dimension
        patches = np.concatenate(patches, axis=0) # shape (B * P, patch_H, patch_W)
        
        return patches
        """

    def __getitem__(self, index):
        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride

        random_dataset = np.random.randint(0, len(self.paths))
        Reynolds_number = self.RN[random_dataset]
        trajectory = self.datasets[random_dataset]

        # Sliding window view implementation: creates large batch sizes
        """
        # Sample a random subset of trajectory
        start_time = np.random.randint(0, trajectory.shape[0] - self.trajectory_length)
        trajectory = trajectory[start_time:start_time + self.trajectory_length, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]
        time_len = len(trajectory) - self.horizon - self.window + 1
        X = np.lib.stride_tricks.sliding_window_view(trajectory, time_len, axis=0) # (horizon + window - 1, lat, lon, time_len)
        X = rearrange(X, "dynamics lat lon time_len -> time_len dynamics 1 lat lon")
        """
        # Simple implementation
        # Sample a random subset of trajectory
        start_time = np.random.randint(0, trajectory.shape[0] - self.window - self.horizon)
        X = trajectory[start_time:start_time + self.window + self.horizon, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]
        X = rearrange(X, "t lat lon -> t 1 lat lon")

        # return {"dynamics": X, "Reynolds_number": Reynolds_number}
        return {"dynamics": X}

    def __len__(self):
        return self.size
