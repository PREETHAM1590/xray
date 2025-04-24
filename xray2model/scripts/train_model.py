# Script for training the 2D-to-3D reconstruction model.

# TODO:
# - Implement realistic input adaptation for DRR -> 3D UNet input in data_loader.py or model.
# - Implement more sophisticated MONAI transforms in data_loader.py.
# - Add learning rate scheduler.
# - Handle potential GPU usage and distributed training if needed.
# - Add option for different model architectures (e.g., VNet, 2D Enc + 3D Dec).
# - Implement evaluation metrics like Chamfer Distance (requires mesh generation, computationally expensive during validation).

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
import glob

# MONAI imports
from monai.losses import DiceLoss, SSIMLoss
from monai.metrics import PSNRMetric, SSIMMetric
from monai.utils import Average, first, ensure_tuple
from monai.visualize import plot_2d_or_3d_image # For TensorBoard logging
# from monai.utils import set_determinism # Optional for reproducibility

# Project imports (adjust relative paths if script structure changes)
try:
    from ..data_loader import get_dataloaders
    from ..models.unet_reconstruction import ReconstructionUNet
except ImportError:
    print("Warning: Could not perform relative imports. Attempting direct import.")
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from data_loader import get_dataloaders
    from models.unet_reconstruction import ReconstructionUNet


# --- Configuration & Hyperparameters (Defaults) ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 2 
NUM_EPOCHS = 100
VAL_INTERVAL = 5 
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
DEFAULT_LOSS = "L1" # Options: L1, MSE, Dice, SSIM

def train(args):
    print("--- Model Training Script ---")
    print(f"DRR Directory: {args.drr_dir}")
    print(f"CT Directory: {args.ct_dir}")
    print(f"Output Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}, Loss: {args.loss_type}")
    print(f"Resume Training: {args.resume}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- Setup ---
    # set_determinism(seed=args.seed) # Optional
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Initialize Model
    # Using MONAI UNet (3D) as default. 
    # Alternatives: VNet, or a custom 2D Encoder + 3D Decoder (would require new model class).
    model = ReconstructionUNet(
        spatial_dims=3,
        in_channels=1, # Assuming 1 channel DRR input after transforms
        out_channels=1, # Assuming 1 channel CT output
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)

    # Initialize Loss Function
    if args.loss_type.upper() == 'L1':
        loss_function = nn.L1Loss()
        print("Using L1 Loss (MAE)")
    elif args.loss_type.upper() == 'MSE':
        loss_function = nn.MSELoss()
        print("Using MSE Loss")
    elif args.loss_type.upper() == 'DICE':
        loss_function = DiceLoss(sigmoid=True, include_background=True, to_onehot_y=False)
        print("Using Dice Loss (Sigmoid activation assumed on model output)")
    elif args.loss_type.upper() == 'SSIM':
        loss_function = SSIMLoss(spatial_dims=3, data_range=1.0) # Assumes data scaled [0,1]
        print("Using SSIM Loss")
    else:
        print(f"Warning: Unknown loss type '{args.loss_type}'. Defaulting to L1 Loss.")
        loss_function = nn.L1Loss()

    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TODO: Add Learning Rate Scheduler

    # --- Checkpoint Loading (Resume Training) ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, "latest_model.pth")
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1 
                best_val_loss = checkpoint.get('best_val_loss', float('inf')) 
                print(f"Loaded model from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                start_epoch = 0
                best_val_loss = float('inf')
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")

    # --- Get DataLoaders ---
    train_loader, val_loader = get_dataloaders(
        drr_dir=args.drr_dir,
        ct_dir=args.ct_dir,
        drr_suffix=args.drr_suffix,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        random_seed=args.seed
    )

    # --- Training Loop ---
    print("\nStarting Training Loop...")
    if not train_loader.dataset: 
        print("Error: Training dataset is empty. Cannot train.")
        writer.close()
        return

    # Initialize Metrics
    psnr_metric = PSNRMetric(max_val=1.0, reduction=Average.MEAN) 
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction=Average.MEAN) 

    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # --- Training Phase ---
        model.train()
        train_epoch_loss = 0
        train_step = 0
        for batch_drr, batch_ct in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            train_step += 1
            optimizer.zero_grad()
            target_tensor = batch_ct.to(device) # Target is the 3D CT volume [B, 1, D, H, W]
            
            # --- Input Adaptation (Placeholder) ---
            # Adapt DRR [B, 1, H, W] to match CT depth for 3D UNet [B, 1, D, H, W]
            # Simple strategy: Repeat the 2D slice along the depth dimension.
            drr_input_2d = batch_drr.to(device)
            target_depth = target_tensor.shape[2] # Get D from CT tensor
            # Unsqueeze to add depth dim: [B, 1, H, W] -> [B, 1, 1, H, W]
            # Repeat along depth dim: [B, 1, 1, H, W] -> [B, 1, D, H, W]
            input_tensor = drr_input_2d.unsqueeze(2).repeat(1, 1, target_depth, 1, 1)
            # --- End Input Adaptation ---

            try:
                outputs = model(input_tensor)
                loss = loss_function(outputs, target_tensor)

                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            except Exception as e:
                 print(f"\nError during training step {train_step}: {e}")
                 print(f"Input shape: {input_tensor.shape}, Target shape: {target_tensor.shape}")
                 continue 

        avg_train_loss = train_epoch_loss / train_step if train_step > 0 else 0
        print(f"\n  Average Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # --- Validation Phase ---
        if (epoch + 1) % args.val_interval == 0 and val_loader.dataset: 
            print("\n  --- Validation ---")
            model.eval()
            val_epoch_loss = 0
            val_step = 0
            psnr_metric.reset()
            ssim_metric.reset()
            
            with torch.no_grad():
                for i, (val_drr, val_ct) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
                    val_step += 1
                    target_tensor = val_ct.to(device)
                    
                    # --- Input Adaptation (Placeholder - same as training) ---
                    drr_input_2d = val_drr.to(device)
                    target_depth = target_tensor.shape[2] # Get D from CT tensor
                    input_tensor = drr_input_2d.unsqueeze(2).repeat(1, 1, target_depth, 1, 1)
                    # --- End Input Adaptation ---

                    try:
                        outputs = model(input_tensor)
                        loss = loss_function(outputs, target_tensor) 
                        val_epoch_loss += loss.item()

                        outputs_clamped = torch.clamp(outputs, 0.0, 1.0) 

                        psnr_metric(y_pred=outputs_clamped, y=target_tensor)
                        ssim_metric(y_pred=outputs_clamped, y=target_tensor)
                        
                        # Log example images to TensorBoard (first batch of validation)
                        if i == 0:
                            # Log middle slice of the first item in the batch
                            mid_slice_idx = target_tensor.shape[2] // 2 # Depth dimension
                            plot_2d_or_3d_image(target_tensor[0, :, mid_slice_idx, :, :], epoch + 1, writer, index=0, tag="Target_CT_Slice")
                            plot_2d_or_3d_image(outputs_clamped[0, :, mid_slice_idx, :, :], epoch + 1, writer, index=0, tag="Output_CT_Slice")
                            # Log DRR input as well (assuming it's 2D [B, C, H, W])
                            # Check ndim before plotting DRR as it might already be adapted
                            if batch_drr.ndim == 4: # Check original batch_drr shape
                                 plot_2d_or_3d_image(batch_drr[0], epoch + 1, writer, index=0, tag="Input_DRR")

                    except Exception as e:
                         print(f"\nError during validation step {val_step}: {e}")
                         continue 

            avg_val_loss = val_epoch_loss / val_step if val_step > 0 else 0
            avg_psnr = psnr_metric.aggregate().item()
            avg_ssim = ssim_metric.aggregate().item()
            
            print(f"  Average Validation Loss: {avg_val_loss:.4f}")
            print(f"  Average Validation PSNR: {avg_psnr:.2f}") 
            print(f"  Average Validation SSIM: {avg_ssim:.4f}") 
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            writer.add_scalar("Metric/Val_PSNR", avg_psnr, epoch) 
            writer.add_scalar("Metric/Val_SSIM", avg_ssim, epoch) 

            # Checkpointing
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'loss_type': args.loss_type,
                }, save_path)
                print(f"  Saved new best model (Epoch {epoch+1}) to {save_path}")
            
            # Save latest checkpoint
            latest_save_path = os.path.join(args.checkpoint_dir, "latest_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss, 
                'loss_type': args.loss_type,
            }, latest_save_path)


        epoch_duration = time.time() - epoch_start_time
        print(f"--- Epoch {epoch + 1} Duration: {epoch_duration:.2f} seconds ---")

    total_duration = time.time() - total_start_time
    print(f"\n--- Training Finished ---")
    print(f"Total Duration: {total_duration/60:.2f} minutes")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D-to-3D Reconstruction Model.")
    parser.add_argument("--drr_dir", type=str, required=True, help="Directory containing input DRR images.")
    parser.add_argument("--ct_dir", type=str, required=True, help="Directory containing corresponding 3D CT scans.")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Directory to save model checkpoints.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save TensorBoard logs.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--val_interval", type=int, default=VAL_INTERVAL, help="Validation frequency (epochs).")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument("--drr_suffix", type=str, default="_drr_axis0.png", help="Suffix used for generated DRR filenames.")
    parser.add_argument("--loss_type", type=str, default=DEFAULT_LOSS, choices=['L1', 'MSE', 'Dice', 'SSIM'], help="Loss function type.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--resume", action='store_true', help="Resume training from latest checkpoint in checkpoint_dir.")
    # TODO: Add arguments for model params, scheduler params, augmentation levels etc.

    args = parser.parse_args()
    
    if not os.path.isdir(args.drr_dir):
         print(f"Error: DRR directory not found: {args.drr_dir}")
         exit()
    if not os.path.isdir(args.ct_dir):
         print(f"Error: CT directory not found: {args.ct_dir}")
         exit()

    train(args)
