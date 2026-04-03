import os
import gc
import time
import math
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from model import CGFformer

def compute_total_loss(output, gt):
    output = output.to(torch.float32)
    gt = gt.to(torch.float32)

    loss_ref = F.l1_loss(output, gt)
    
    return loss_ref

# =========================================================================
# 指数衰减学习率函数：每个 epoch 都在平滑下降，在第 90 个 epoch 时刚好减半
# =========================================================================
def learning_rate_function(epoch, lr_max):
    return lr_max * math.exp((-1) * (math.log(2)) / 90 * epoch)


def main():
    torch.cuda.empty_cache()

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    
    start_time = time.time()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"Using device: {device} with {gpu_count} GPUs")

    train_set_path = "/root/autodl-tmp/Dataset/WV3/WorldView3/training_wv3/train_wv3.h5"
    valid_set_path = "/root/autodl-tmp/Dataset/WV3/WorldView3/training_wv3/valid_wv3.h5"
    checkpoint_save_dir = "/root/CGFformer/weights/"
    
    resume_weight_path = "" 
    
    task = "wv3"             
    epochs = 400             
    lr_max = 0.0006          
    ckpt_interval = 20      
    feat_dim = 32            
    ratio = 2047.0           

    batch_size = 16       

    os.makedirs(checkpoint_save_dir, exist_ok=True)

    if task == "wv3":
        pan_channels, lms_channels = 1, 8
    elif task in ["qb", "gf2"]:
        pan_channels, lms_channels = 1, 4
    else:
        raise ValueError("不支持的 Task 类型。")

    print(f"Loading training data from {train_set_path}...")
    with h5py.File(train_set_path, 'r') as f:
        gt = torch.from_numpy(np.array(f['gt'][:], dtype=np.float32))
        pan = torch.from_numpy(np.array(f['pan'][:], dtype=np.float32))
        ms = torch.from_numpy(np.array(f['ms'][:], dtype=np.float32))
        lms = torch.from_numpy(np.array(f['lms'][:], dtype=np.float32))
        
    print(f"Loading validation data from {valid_set_path}...")
    with h5py.File(valid_set_path, 'r') as f:
        val_gt = torch.from_numpy(np.array(f['gt'][:], dtype=np.float32))
        val_pan = torch.from_numpy(np.array(f['pan'][:], dtype=np.float32))
        val_ms = torch.from_numpy(np.array(f['ms'][:], dtype=np.float32))
        val_lms = torch.from_numpy(np.array(f['lms'][:], dtype=np.float32))

    train_ds = TensorDataset(pan / ratio, gt / ratio, ms / ratio, lms / ratio)
    val_ds = TensorDataset(val_pan / ratio, val_gt / ratio, val_ms / ratio, val_lms / ratio)
    
    num_workers = min(12, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print("Initializing CGFformer model...")
    model = CGFformer(pan_channels=pan_channels, lms_channels=lms_channels, feat_dim=feat_dim)
    model.to(device)

    if gpu_count > 1:
        print(f"Activating DataParallel on {gpu_count} GPUs!")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr_max, 
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # 默认初始状态
    start_epoch = 1
    best_val_loss = float('inf')
    
    if resume_weight_path and os.path.isfile(resume_weight_path):
        print(f"\n=> 正在加载预训练权重: '{resume_weight_path}'")
        checkpoint = torch.load(resume_weight_path, map_location=device)
        
        # 加载模型权重
        if isinstance(model, nn.DataParallel):
            if not list(checkpoint['model'].keys())[0].startswith('module.'):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
        else:
            if list(checkpoint['model'].keys())[0].startswith('module.'):
                new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['model'])
                
        # 加载优化器状态（此时旧的学习率被带入）
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # =========================================================================
        # 强制清除过去的训练日志：重置 epoch 为 1，清空 val_loss 记录
        # 这确保了即将进入 for 循环时，学习率会被 learning_rate_function(1) 强制覆盖回初始高点
        # =========================================================================
        start_epoch = 1
        best_val_loss = float('inf')
        
        print(f"=> 权重加载成功！训练日志已被清空。学习率即将被重置，从第 1 个 epoch 重新开始训练。\n")

    print("=" * 60)
    print(f"Starting training (Pure L1 Loss) | Batch Size: {batch_size}...")
    print("=" * 60)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs + 1):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        
        # =====================================================================
        # 每一轮开始时，直接计算当前 epoch 对应的学习率，并强行覆盖优化器里的旧值
        # =====================================================================
        current_lr = learning_rate_function(epoch, lr_max)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        model.train()
        epoch_train_loss = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [Train]", ncols=100, leave=True)
        for batch_idx, (pan_batch, gt_batch, ms_batch, lms_batch) in enumerate(pbar):
            pan_batch = pan_batch.to(device, dtype=torch.float32)
            gt_batch = gt_batch.to(device, dtype=torch.float32)
            ms_batch = ms_batch.to(device, dtype=torch.float32)
            
            # 模型推理
            output, H_E, L_E, H_FG, L_FG, H_out, L_out = model(pan_batch, ms_batch)

            total_loss = compute_total_loss(output, gt_batch)
            
            if torch.isnan(total_loss).any():
                print(f"NaN detected at epoch {epoch}! Skipping this batch.")
                continue
                
            total_loss.backward()
            epoch_train_loss.append(total_loss.item())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 

            pbar.set_postfix({'loss': f'{total_loss.item():.6f}'})

        t_loss = np.mean(epoch_train_loss)

        torch.cuda.empty_cache() 
        model.eval()
        epoch_valid_loss = []
        with torch.no_grad():
            valid_pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [Valid]", ncols=100, leave=True)
            for pan_batch, gt_batch, ms_batch, lms_batch in valid_pbar:
                pan_batch = pan_batch.to(device, dtype=torch.float32)
                gt_batch = gt_batch.to(device, dtype=torch.float32)
                ms_batch = ms_batch.to(device, dtype=torch.float32)

                output, H_E, L_E, H_FG, L_FG, H_out, L_out = model(pan_batch, ms_batch)
                loss = compute_total_loss(output, gt_batch)
                
                epoch_valid_loss.append(loss.item())
                valid_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        v_loss = np.mean(epoch_valid_loss)
        
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch [{epoch:3d}/{epochs}] | Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        
        with open("loss.txt", "a") as f:
            f.write(f"epoch: {epoch} | train_loss: {t_loss:.6f} | valid_loss: {v_loss:.6f}\n")

        model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint_data = {
            'model': model_state_to_save,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(checkpoint_data, os.path.join(checkpoint_save_dir, "CGFformer_best.pth"))
            print(f"  → Best model saved (Val Loss: {best_val_loss:.6f})")

        if epoch % ckpt_interval == 0 or epoch == epochs:
            torch.save(checkpoint_data, os.path.join(checkpoint_save_dir, f"CGFformer_epoch_{epoch}.pth"))
            print(f"  → Checkpoint saved at epoch {epoch}")

if __name__ == "__main__":
    main()