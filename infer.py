import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange
import numbers

import os
import argparse
import yaml
from PIL import Image
import time
import numpy as np
from pathlib import Path

from train import RetinexFormer

# ----------Utility Functions----------
def pad_image_to_multiple(img_tensor, multiple=4):
    """Pad image to be divisible by multiple (4 for RetinexFormer)."""
    _, h, w = img_tensor.shape
    
    # Calculate padding needed
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    # Pad symmetrically
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding
    img_padded = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    
    padding = (pad_top, pad_bottom, pad_left, pad_right)
    return img_padded, padding

def unpad_image(img_tensor, original_size, padding):
    """Remove padding to restore original size."""
    pad_top, pad_bottom, pad_left, pad_right = padding
    h, w = original_size
    
    # Remove padding
    if pad_bottom > 0:
        img_tensor = img_tensor[:, :-pad_bottom, :]
    if pad_top > 0:
        img_tensor = img_tensor[:, pad_top:, :]
    if pad_right > 0:
        img_tensor = img_tensor[:, :, :-pad_right]
    if pad_left > 0:
        img_tensor = img_tensor[:, :, pad_left:]
    
    return img_tensor

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    # Clamp values to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)

# ----------Model----------
def load_model(model_path, model_cfg, device):
    """Load trained model from checkpoint."""
    # Create model with same architecture as training
    model_type = model_cfg.pop('type', 'RetinexFormer')
    if model_type == 'RetinexFormer':
        model = RetinexFormer(**model_cfg)  # Now model_cfg doesn't contain 'type'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model weights
    if model_path.endswith('.pt'):
        # Load only state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Load full checkpoint (for compatibility)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

# ----------Dataset----------
def get_inference_transform():
    """Create minimal transforms for inference - only normalization and tensor conversion."""
    # For inference, we only need normalization (if used during training) and tensor conversion
    # No resizing, cropping, or augmentation
    transform_list = []
    
    # Convert to tensor first
    transform_list.append(ToTensorV2())
    
    return A.Compose(transform_list)

class InferenceDataset(Dataset):
    """
    Dataset for inference on single images or directories.
    No target images needed for inference.
    """
    
    def __init__(self, input_path, transform=None):
        self.input_path = input_path
        self.transform = transform
        
        # Check if input is a file or directory
        if os.path.isfile(input_path):
            self.image_files = [os.path.basename(input_path)]
            self.input_dir = os.path.dirname(input_path)
        elif os.path.isdir(input_path):
            self.input_dir = input_path
            self.image_files = self._get_image_files()
        else:
            raise ValueError(f"Input path not found: {input_path}")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {input_path}")
    
    def _get_image_files(self):
        """Get list of image files from input directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in os.listdir(self.input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        
        # Sort for consistent ordering
        image_files.sort()
        return image_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Load input image for inference."""
        filename = self.image_files[idx]
        input_img_path = os.path.join(self.input_dir, filename)
        
        # Load image using PIL
        img = Image.open(input_img_path).convert('RGB')
        
        # Convert to tensor and normalize to [0, 1]
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(img)
        
        # Apply transforms if available (minimal processing)
        if self.transform is not None:
            # Convert tensor back to numpy for albumentations
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            # Apply transforms
            transformed = self.transform(image=img_np)
            img_tensor = transformed['image']
        
        return img_tensor, filename

def inference_single_image(model, img_tensor, device):
    """Perform inference on a single image."""
    model.eval()
    
    with torch.no_grad():
        # Store original size
        original_size = img_tensor.shape[1:]
        
        # Pad image to be divisible by 4 (required for RetinexFormer)
        img_padded, padding = pad_image_to_multiple(img_tensor, multiple=4)
        
        # Add batch dimension
        img_batch = img_padded.unsqueeze(0).to(device)
        
        # Perform inference
        output_batch = model(img_batch)
        
        # Remove batch dimension and move back to CPU
        output_tensor = output_batch.squeeze(0).cpu()
        
        # Remove padding to restore original size
        output_tensor = unpad_image(output_tensor, original_size, padding)
        
        return output_tensor

def main(args):
    # Load config from YAML file
    cfg_path = args.cfg
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Extract configurations
    model_cfg = cfg.get('model', {})
    val_cfg = cfg.get('val', {})
    
    # Get checkpoint path from config
    ckpt_path = val_cfg.get('ckpt_path')
    if not ckpt_path:
        raise ValueError("ckpt_path not found in val section of config")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    # Get input path from config
    input_path = val_cfg.get('input')
    if not input_path:
        raise ValueError("input path not found in val section of config")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # Create output directory structure
    # Extract checkpoint name and create organized output dir
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]  # checkpoint_epoch_10
    config_name = os.path.splitext(os.path.basename(cfg_path))[0]  # Rain13K
    val_dir_name = os.path.basename(input_path)  # val_tiny
    
    # Get the parent directory of checkpoint (logs/restormer_turbidity_20250917_000725)
    ckpt_parent_dir = os.path.dirname(ckpt_path)
    
    # Create output directory: logs/restormer_turbidity_20250917_000725/checkpoint_epoch_10_Rain13K_val_tiny/
    output_dir = os.path.join(ckpt_parent_dir, f"{ckpt_name}_{config_name}_{val_dir_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(ckpt_path, model_cfg, device)
    print(f"Model loaded successfully")

    # Create inference dataset
    transform = get_inference_transform()
    dataset = InferenceDataset(input_path, transform=transform)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=val_cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=val_cfg.get('num_workers', 4),
        pin_memory=val_cfg.get('pin_memory', True),
        drop_last=False
    )
    print(f"Starting inference on {len(dataset)} images...")

    for batch_idx, (img_tensor, filename) in enumerate(dataloader):
        # Remove batch dimension from dataloader
        img_tensor = img_tensor.squeeze(0)
        
        # Extract filename from tuple/list if needed
        if isinstance(filename, (tuple, list)):
            filename = filename[0]
        
        print(f"Processing {filename} ({batch_idx + 1}/{len(dataset)})")
        
        # Perform inference
        output_tensor = inference_single_image(model, img_tensor, device)
        
        # Convert to PIL and save
        output_img = tensor_to_pil(output_tensor)
        
        # Generate output filename (keep same name as original)
        output_filename = filename
        output_path = os.path.join(output_dir, output_filename)
        
        # Save image
        output_img.save(output_path)
        print(f"Saved: {output_path}")
    
    print(f"Inference completed! {len(dataset)} images processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference with RetinexFormer model by refactoring the original code to remove the basicsr dependency")
    parser.add_argument('--cfg', type=str, required=True, 
                       help='Path to the YAML config file used for training.')
    
    args = parser.parse_args()
    main(args)