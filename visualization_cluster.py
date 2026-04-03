import os
import re
import torch
import torch.nn.functional as F
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import FSGformer

# ================= Configuration =================
CONFIG = {
    "test_set_path": "test_wv3_multiExm1.h5",
    "weights_dir": "weights/worldview3/",
    "save_dir": "visualization_results",
    "batch_size": 1,    # 保持为1，方便逐张处理和保存
    "task": "wv3",      # 用于确定通道数
    "cluster_num": 32   # Canfilter中的聚类数量
}
# =================================================

class DataSet(torch.utils.data.Dataset):
    """
    Dataset loader derived from test.py
    """
    def __init__(self, file_path):
        super(DataSet, self).__init__()
        # Use 'r' mode for safety
        data = h5py.File(file_path, 'r') 

        # Low resolution MS
        lms1 = data["lms"][...]
        lms1 = np.array(lms1, dtype=np.float32) / 2047
        self.lms = torch.from_numpy(lms1)

        # MS (High-Pass filtered typically, based on test.py logic)
        ms1 = data["ms"][...]
        ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047
        ms1_tmp = self.get_edge(ms1)
        self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2)

        # PAN High-Pass
        pan1 = data['pan'][...]
        pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047
        pan1 = np.squeeze(pan1, axis=3)
        pan_hp_tmp = self.get_edge(pan1)
        pan_hp_tmp = np.expand_dims(pan_hp_tmp, axis=3)
        self.pan_hp = torch.from_numpy(pan_hp_tmp).permute(0, 3, 1, 2)

        # PAN Original
        pan1 = data['pan'][...]
        pan1 = np.array(pan1, dtype=np.float32) / 2047
        self.pan = torch.from_numpy(pan1)

    def get_edge(self, data):
        rs = np.zeros_like(data)
        N = data.shape[0]
        for i in range(N):
            if len(data.shape) == 3:
                rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
            else:
                rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
        return rs

    def __getitem__(self, index):
        # Return: LMS(original low res), MS_HP(input to model), PAN_HP, PAN(original)
        return self.lms[index, :, :, :].float(), \
            self.ms_hp[index, :, :, :].float(), \
            self.pan_hp[index, :, :, :].float(), \
            self.pan[index, :, :, :].float()

    def __len__(self):
        return self.lms.shape[0]

def get_sorted_checkpoints(weight_dir):
    """Finds and sorts all checkpoint files by epoch number."""
    if not os.path.exists(weight_dir):
        print(f"Error: Weights directory {weight_dir} does not exist.")
        return []
    
    files = os.listdir(weight_dir)
    checkpoints = []
    pattern = re.compile(r"checkpoint_(\d+)\.pth")
    
    for f in files:
        match = pattern.search(f)
        if match:
            epoch = int(match.group(1))
            full_path = os.path.join(weight_dir, f)
            checkpoints.append((epoch, full_path))
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def save_heatmap(indices, shape, save_path, title):
    """
    Saves the clustering indices as a color-mapped image.
    """
    indices = indices.view(shape).cpu().numpy()
    plt.figure(figsize=(8, 8))
    # 'tab20' is good for categorical data up to 20 classes. 
    # For 32 classes, colors will repeat or we can use 'nipy_spectral'
    plt.imshow(indices, cmap='nipy_spectral', interpolation='nearest', vmin=0, vmax=CONFIG['cluster_num']-1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_image(tensor, save_path, mode='gray'):
    """
    Saves a tensor as a standard image.
    tensor: (C, H, W)
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0) # (H, W, C)
    
    # Normalize to 0-1 if not already
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(8, 8))
    if mode == 'gray':
        plt.imshow(img[:, :, 0], cmap='gray')
    elif mode == 'rgb':
        # If more than 3 channels, take first 3 or specific bands
        # For WV3 (8 bands), RGB is often band 5, 3, 2 (indices 4, 2, 1)
        # Here we use indices 0, 1, 2 for simplicity unless task logic is added
        if img.shape[2] >= 3:
            # Simple RGB visualization using first 3 bands
            plt.imshow(img[:, :, 0:3]) 
        else:
            plt.imshow(img[:, :, 0], cmap='gray')
            
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

# Global hooks storage
hook_data = {
    'pan_indices': [],
    'ms_indices': []
}

def hook_pan(module, input, output):
    # output: (low_freq, high_freq, cluster_indice)
    # cluster_indice shape: (B, H*W)
    hook_data['pan_indices'] = output[2]

def hook_ms(module, input, output):
    hook_data['ms_indices'] = output[2]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Setup Data
    print(f"Loading dataset from {CONFIG['test_set_path']}...")
    dataset = DataSet(CONFIG['test_set_path'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    total_samples = len(dataset)
    print(f"Total samples found: {total_samples}")

    # 2. Save Original Images (First Pass)
    # We do this first so we have the base images even if we don't run all epochs
    print("Step 1: Saving original images for all samples...")
    for idx, batch in enumerate(tqdm(dataloader, desc="Saving Originals")):
        lms, ms_hp, pan_hp, pan = batch
        
        # Create directory for this sample
        sample_dir = os.path.join(CONFIG['save_dir'], f"sample_{idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save PAN Original
        save_image(pan[0], os.path.join(sample_dir, "original_pan.png"), mode='gray')
        
        # Save MS Upsampled Original
        # Upsample lms (Low Res) to match PAN size using Bilinear
        # lms shape: (B, C, H/4, W/4) -> (B, C, H, W)
        ms_upsampled = F.interpolate(lms, scale_factor=4, mode='bilinear', align_corners=False)
        save_image(ms_upsampled[0], os.path.join(sample_dir, "original_ms_upsampled.png"), mode='rgb')

    # 3. Model Setup
    if CONFIG["task"] == "wv3":
        pan_channels, lms_channels = 1, 8
    elif CONFIG["task"] in ["qb", "gf2"]:
        pan_channels, lms_channels = 1, 4
    
    model = FSGformer(pan_channels, lms_channels).to(device)
    
    # Register Hooks
    model.lfs.canfilter_pan.register_forward_hook(hook_pan)
    model.lfs.canfilter_ms.register_forward_hook(hook_ms)

    # 4. Iterate Checkpoints (Epochs)
    checkpoints = get_sorted_checkpoints(CONFIG['weights_dir'])
    print(f"Found {len(checkpoints)} checkpoints. Starting visualization loop...")

    for epoch, ckpt_path in tqdm(checkpoints, desc="Epochs"):
        # Load Weights
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Skipping {ckpt_path}: {e}")
            continue

        model.eval()
        
        # Run inference on all samples for this epoch
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                lms, ms_hp, pan_hp, pan = batch
                
                # Move to device
                pan = pan.to(device)
                ms_hp = ms_hp.to(device) # Model expects 'ms' argument as the HP filtered version
                
                # Forward pass triggers hooks
                _ = model(pan, ms_hp)
                
                # Retrieve indices
                # shape: (B, H*W) -> here B=1, so (1, H*W)
                pan_idx = hook_data['pan_indices']
                ms_idx = hook_data['ms_indices']
                
                # Get dimensions for reshaping
                # pan shape (B, 1, H, W)
                B, C, H, W = pan.shape
                
                sample_dir = os.path.join(CONFIG['save_dir'], f"sample_{idx:03d}")
                
                # Save PAN Cluster
                if pan_idx is not None:
                    save_name = os.path.join(sample_dir, f"epoch_{epoch}_pan_cluster.png")
                    save_heatmap(pan_idx[0], (H, W), save_name, f"PAN Cluster - Epoch {epoch}")
                
                # Save MS Cluster
                # MS cluster map size matches PAN because LFS upsamples MS before clustering
                if ms_idx is not None:
                    save_name = os.path.join(sample_dir, f"epoch_{epoch}_ms_cluster.png")
                    save_heatmap(ms_idx[0], (H, W), save_name, f"MS Cluster - Epoch {epoch}")

                # Reset hooks storage for next batch
                hook_data['pan_indices'] = None
                hook_data['ms_indices'] = None

    print(f"\nProcessing complete. Results saved in {CONFIG['save_dir']}")

if __name__ == "__main__":
    main()