import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange
import numbers

import os
import argparse
import yaml
import logging
from datetime import datetime
import math
from PIL import Image
import warnings
import time

# ----------Logging Setup----------
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\x1b[38;20m',      # grey
        'INFO': '\x1b[36;20m',       # cyan
        'WARNING': '\x1b[33;20m',    # yellow
        'ERROR': '\x1b[31;20m',      # red
        'CRITICAL': '\x1b[31;1m',    # bold red
    }
    RESET = '\x1b[0m'

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, '')
        original_msg = super().format(record)
        return f"{level_color}{original_msg}{self.RESET}"

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers (if re-running in notebooks)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (detailed, no colors)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (colored, concise)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s %(levelname)s %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# ----------Dataset----------
def get_train_transform(cfg):
    """Create training transforms based on configuration."""
    transform_cfg = cfg.get('transform', {})
    
    transform_list = []
    
    # Resize if specified - this will apply to BOTH input and target
    resize_h = transform_cfg.get('resize_h')
    resize_w = transform_cfg.get('resize_w')
    if resize_h is not None and resize_w is not None:
        transform_list.append(A.Resize(height=resize_h, width=resize_w))
    
    # Random crop if specified - this will apply to BOTH input and target
    crop_size = transform_cfg.get('crop_size')
    if crop_size is not None:
        transform_list.append(A.RandomCrop(height=crop_size[0], width=crop_size[1]))
    
    # Horizontal flip if specified - this will apply to BOTH input and target
    flip_prob = transform_cfg.get('horizontal_flip_prob')
    if flip_prob is not None and flip_prob > 0:
        transform_list.append(A.HorizontalFlip(p=flip_prob))
    
    # Brightness/contrast adjustment if specified - this will apply to BOTH input and target
    brightness_contrast_prob = transform_cfg.get('brightness_contrast_prob')
    if brightness_contrast_prob is not None and brightness_contrast_prob > 0:
        transform_list.append(A.RandomBrightnessContrast(p=brightness_contrast_prob))
    
    # Normalization if specified - this will apply to BOTH input and target
    normalize_mean = transform_cfg.get('normalize_mean')
    normalize_std = transform_cfg.get('normalize_std')
    if normalize_mean is not None and normalize_std is not None:
        transform_list.append(A.Normalize(mean=normalize_mean, std=normalize_std, max_pixel_value=1.0))
    
    # Convert to tensor - this will apply to BOTH input and target
    transform_list.append(ToTensorV2())
    
    return A.Compose(transform_list)

def get_val_transform(cfg):
    """Create validation transforms - minimal processing only."""
    # For validation, we typically want minimal or no transforms
    # to preserve original image characteristics for accurate evaluation
    
    transform_list = []
    
    # Resize if specified - this will apply to BOTH input and target
    transform_cfg = cfg.get('transform', {})
    resize_h = transform_cfg.get('resize_h')
    resize_w = transform_cfg.get('resize_w')
    if resize_h is not None and resize_w is not None:
        transform_list.append(A.Resize(height=resize_h, width=resize_w))
    
    # Only add normalization if explicitly specified in config
    normalize_mean = transform_cfg.get('normalize_mean')
    normalize_std = transform_cfg.get('normalize_std')
    
    if normalize_mean is not None and normalize_std is not None:
        transform_list.append(A.Normalize(mean=normalize_mean, std=normalize_std, max_pixel_value=1.0))
    
    # Convert to tensor
    transform_list.append(ToTensorV2())
    
    return A.Compose(transform_list)

class ImageRestorationDataset(Dataset):
    """
    Dataset for image restoration tasks.
    Accepts separate input and target paths.
    """
    
    def __init__(self, input_path, target_path, transform=None):
        self.input_path = input_path
        self.target_path = target_path
        self.transform = transform
        
        # Check if directories exist
        if not os.path.exists(self.input_path):
            raise ValueError(f"Input directory not found: {self.input_path}")
        if not os.path.exists(self.target_path):
            raise ValueError(f"Target directory not found: {self.target_path}")
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.input_path}")
        
        print(f"Found {len(self.image_files)} images in {self.input_path}")
    
    def _get_image_files(self):
        """Get list of image files from input directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in os.listdir(self.input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        
        # Sort for consistent ordering
        image_files.sort()
        return image_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Load input and target image pair."""
        filename = self.image_files[idx]
        
        # Load input image
        input_img_path = os.path.join(self.input_path, filename)
        input_img = self._load_image(input_img_path)
        
        # Load target image (same filename)
        target_img_path = os.path.join(self.target_path, filename)
        target_img = self._load_image(target_img_path)
        
        # Apply transforms if available
        if self.transform is not None:
            # Convert tensors back to numpy for albumentations
            input_img_np = input_img.permute(1, 2, 0).numpy()
            target_img_np = target_img.permute(1, 2, 0).numpy()
            
            # Apply transforms to input
            transformed_input = self.transform(image=input_img_np)
            input_img = transformed_input['image']
            
            # Apply the same spatial transforms to target (resize, crop, flip)
            transform_cfg = self.transform.transforms
            target_transforms = []
            
            # Extract only spatial transforms (resize, crop, flip)
            for transform in transform_cfg:
                if isinstance(transform, (A.Resize, A.RandomCrop, A.HorizontalFlip)):
                    target_transforms.append(transform)
            
            # Apply spatial transforms to target
            if target_transforms:
                target_transform = A.Compose(target_transforms)
                transformed_target = target_transform(image=target_img_np)
                target_img_np = transformed_target['image']
            
            # Convert target to tensor and ensure correct format
            target_img = torch.from_numpy(target_img_np).float()
            if target_img.dim() == 3 and target_img.shape[0] != 3:
                target_img = target_img.permute(2, 0, 1)
        else:
            # No transforms applied, ensure tensors are in correct format
            if input_img.dim() == 3 and input_img.shape[0] != 3:
                input_img = input_img.permute(2, 0, 1)
            if target_img.dim() == 3 and target_img.shape[0] != 3:
                target_img = target_img.permute(2, 0, 1)
        
        # Debug: print final tensor shapes
        # print(f"DEBUG: {filename} - Final shapes - Input: {input_img.shape}, Target: {target_img.shape}")
        
        # Check if dimensions match expected size from config
        # Get expected size from transform config
        transform_cfg = self.transform.transforms if self.transform else []
        expected_h = None
        expected_w = None
        
        # Find resize dimensions from transforms
        for transform in transform_cfg:
            if isinstance(transform, A.Resize):
                expected_h = transform.height
                expected_w = transform.width
                break
        
        # Default fallback if no resize specified
        if expected_h is None or expected_w is None:
            expected_h = 256  # Default divisible by 4 for RetinexFormer
            expected_w = 256  # Default divisible by 4 for RetinexFormer
        if input_img.shape[1] != expected_h or input_img.shape[2] != expected_w:
            print(f"WARNING: Input image dimensions {input_img.shape[1]}x{input_img.shape[2]} don't match expected {expected_h}x{expected_w}")
        if target_img.shape[1] != expected_h or target_img.shape[2] != expected_w:
            print(f"WARNING: Target image dimensions {target_img.shape[1]}x{target_img.shape[2]} don't match expected {expected_h}x{expected_w}")
        
        # Fallback: if dimensions don't match expected size, force resize
        if input_img.shape[1] != expected_h or input_img.shape[2] != expected_w:
            print(f"FALLBACK: Resizing input from {input_img.shape} to ({3}, {expected_h}, {expected_w})")
            input_img = F.interpolate(input_img.unsqueeze(0), size=(expected_h, expected_w), mode='bilinear', align_corners=False).squeeze(0)
        
        if target_img.shape[1] != expected_h or target_img.shape[2] != expected_w:
            print(f"FALLBACK: Resizing target from {target_img.shape} to ({3}, {expected_h}, {expected_w})")
            target_img = F.interpolate(target_img.unsqueeze(0), size=(expected_h, expected_w), mode='bilinear', align_corners=False).squeeze(0)
        
        # Final check: ensure both images have the same shape
        if input_img.shape != target_img.shape:
            raise ValueError(f"Input and target shapes don't match: input {input_img.shape} vs target {target_img.shape}")
        
        # Additional validation: ensure tensors are in correct format
        if input_img.dim() != 3 or target_img.dim() != 3:
            raise ValueError(f"Images must be 3D tensors (C, H, W), got input: {input_img.dim()}D, target: {target_img.dim()}D")
        
        if input_img.shape[0] != 3 or target_img.shape[0] != 3:
            raise ValueError(f"Images must have 3 channels, got input: {input_img.shape[0]}, target: {target_img.shape[0]}")
        
        return input_img, target_img
    
    def _load_image(self, image_path):
        """Load and preprocess image."""
        try:
            # Load image using PIL
            img = Image.open(image_path).convert('RGB')
            
            # Convert to tensor and normalize to [0, 1]
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            img_tensor = transform(img)
            return img_tensor
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 256, 256)

# ----------Model----------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img

class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1]):
        super(RetinexFormer, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out

# ----------Loss Functions----------
def l1_loss(pred, target):
    """L1 loss without reduction."""
    return F.l1_loss(pred, target, reduction='none')

def mse_loss(pred, target):
    """MSE loss without reduction."""
    return F.mse_loss(pred, target, reduction='none')

def charbonnier_loss(pred, target, eps=1e-3):
    """Charbonnier loss without reduction."""
    diff = pred - target
    return torch.sqrt((diff * diff) + (eps * eps))

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss

class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss

class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-3):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt((diff * diff) + (self.eps * self.eps))
        loss = loss.mean()  # Apply mean reduction
        return self.loss_weight * loss

def setup_loss(train_cfg):
    """Setup loss function based on configuration."""
    loss_cfg = train_cfg.get('loss', {})
    loss_type = loss_cfg.get('type', 'L1Loss')
    
    if loss_type == 'L1Loss':
        loss_weight = loss_cfg.get('loss_weight', 1.0)
        reduction = loss_cfg.get('reduction', 'mean')
        criterion = L1Loss(loss_weight=loss_weight, reduction=reduction)
        print(f'Loss {loss_type} created with weight={loss_weight}, reduction={reduction}')
    
    elif loss_type == 'MSELoss':
        loss_weight = loss_cfg.get('loss_weight', 1.0)
        reduction = loss_cfg.get('reduction', 'mean')
        criterion = MSELoss(loss_weight=loss_weight, reduction=reduction)
        print(f'Loss {loss_type} created with weight={loss_weight}, reduction={reduction}')
    
    elif loss_type == 'CharbonnierLoss':
        loss_weight = loss_cfg.get('loss_weight', 1.0)
        eps = loss_cfg.get('eps', 1e-3)
        criterion = CharbonnierLoss(loss_weight=loss_weight, eps=eps)
        print(f'Loss {loss_type} created with weight={loss_weight}, eps={eps}')
    
    else:
        print(f'Loss {loss_type} not implemented, using default L1Loss')
        criterion = L1Loss()
    
    return criterion

# ----------Optimizer----------
def setup_optimizer(model, train_cfg):
    """Setup optimizer based on configuration."""
    optim_params = []
    
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            print(f'Params {k} will not be optimized.')
    
    optimizer_cfg = train_cfg.get('optimizer', {})
    optim_type = optimizer_cfg.get('type', 'AdamW')
    lr = optimizer_cfg.get('lr', 1e-4)
    weight_decay = optimizer_cfg.get('weight_decay', 1e-4)
    
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {optim_type} is not supported yet.')
    
    print(f'Optimizer {optim_type} created with lr={lr}, weight_decay={weight_decay}')
    return optimizer

# ----------Scheduler----------
def setup_scheduler(optimizer, train_cfg):
    """Setup learning rate scheduler based on configuration."""
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'StepLR')
    
    if scheduler_type == 'StepLR':
        step_size = scheduler_cfg.get('step_size', 30)
        gamma = scheduler_cfg.get('gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f'Scheduler {scheduler_type} created with step_size={step_size}, gamma={gamma}')
    
    elif scheduler_type == 'CosineAnnealingLR':
        T_max = scheduler_cfg.get('T_max', train_cfg.get('num_epochs', 100))
        eta_min = scheduler_cfg.get('eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        print(f'Scheduler {scheduler_type} created with T_max={T_max}, eta_min={eta_min}')
    
    elif scheduler_type == 'MultiStepLR':
        milestones = scheduler_cfg.get('milestones', [30, 60, 90])
        gamma = scheduler_cfg.get('gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        print(f'Scheduler {scheduler_type} created with milestones={milestones}, gamma={gamma}')
    
    else:
        print(f'Scheduler {scheduler_type} not implemented, using default StepLR')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    return scheduler

# ----------Training Functions----------
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, current_iter):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    l1_losses = 0.0
    mse_losses = 0.0
    charbonnier_losses = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (input_imgs, target_imgs) in enumerate(train_loader):
        # Move data to device
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)
        
        # Debug: print tensor dimensions
        # print(f"DEBUG: Batch {batch_idx} - Input shape: {input_imgs.shape}, Target shape: {target_imgs.shape}")
        
        # Forward pass
        optimizer.zero_grad()
        output_imgs = model(input_imgs)
        
        # Calculate individual losses
        l1_loss_val = F.l1_loss(output_imgs, target_imgs)
        mse_loss_val = F.mse_loss(output_imgs, target_imgs)
        charbonnier_loss_val = torch.sqrt((output_imgs - target_imgs) ** 2 + 1e-3 ** 2).mean()
        
        # Calculate main loss (using the configured criterion)
        loss = criterion(output_imgs, target_imgs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate all losses
        total_loss += loss.item()
        l1_losses += l1_loss_val.item()
        mse_losses += mse_loss_val.item()
        charbonnier_losses += charbonnier_loss_val.item()
        
        # Update iteration counter
        current_iter += 1
        
        # Log progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, "
                       f"Iter {current_iter}, Total: {loss.item():.6f}, "
                       f"L1: {l1_loss_val.item():.6f}, MSE: {mse_loss_val.item():.6f}, "
                       f"Charb: {charbonnier_loss_val.item():.6f}")
    
    # Calculate average losses for the epoch
    avg_total_loss = total_loss / num_batches
    avg_l1_loss = l1_losses / num_batches
    avg_mse_loss = mse_losses / num_batches
    avg_charbonnier_loss = charbonnier_losses / num_batches
    
    return {
        'total_loss': total_loss,
        'avg_total_loss': avg_total_loss,
        'l1_loss': l1_losses,
        'avg_l1_loss': avg_l1_loss,
        'mse_loss': mse_losses,
        'avg_mse_loss': avg_mse_loss,
        'charbonnier_loss': charbonnier_losses,
        'avg_charbonnier_loss': avg_charbonnier_loss
    }, current_iter

def validate_one_epoch(model, val_loader, criterion, device, epoch, logger, current_iter):
    """Validate one epoch."""
    model.eval()
    total_loss = 0.0
    l1_losses = 0.0
    mse_losses = 0.0
    charbonnier_losses = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (input_imgs, target_imgs) in enumerate(val_loader):
            # Move data to device
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Debug: print tensor dimensions
            # print(f"DEBUG: Val Batch {batch_idx} - Input shape: {input_imgs.shape}, Target shape: {target_imgs.shape}")
            
            # Forward pass
            output_imgs = model(input_imgs)
            
            # Calculate individual losses
            l1_loss_val = F.l1_loss(output_imgs, target_imgs)
            mse_loss_val = F.mse_loss(output_imgs, target_imgs)
            charbonnier_loss_val = torch.sqrt((output_imgs - target_imgs) ** 2 + 1e-3 ** 2).mean()
            
            # Calculate main loss (using the configured criterion)
            loss = criterion(output_imgs, target_imgs)
            
            # Accumulate all losses
            total_loss += loss.item()
            l1_losses += l1_loss_val.item()
            mse_losses += mse_loss_val.item()
            charbonnier_losses += charbonnier_loss_val.item()
    
    # Calculate average losses for the epoch
    avg_total_loss = total_loss / num_batches
    avg_l1_loss = l1_losses / num_batches
    avg_mse_loss = mse_losses / num_batches
    avg_charbonnier_loss = charbonnier_losses / num_batches
    
    return {
        'total_loss': total_loss,
        'avg_total_loss': avg_total_loss,
        'l1_loss': l1_losses,
        'avg_l1_loss': avg_l1_loss,
        'mse_loss': mse_losses,
        'avg_mse_loss': avg_mse_loss,
        'charbonnier_loss': charbonnier_losses,
        'avg_charbonnier_loss': avg_charbonnier_loss
    }, current_iter

def main(args):
    # Load config from YAML file
    cfg_path = args.cfg
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # Extract training, validation, and model configurations
    train_cfg = cfg.get('train', {})
    val_cfg = cfg.get('val', {}) 
    model_cfg = cfg.get('model', {})

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"{train_cfg.get('name', 'restormer')}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logger = setup_logging(log_file)

    # Save the loaded config for reference
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create training dataset and dataloader
    train_transform = get_train_transform(train_cfg)
    train_dataset = ImageRestorationDataset(train_cfg['input'], train_cfg['target'], transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 1),
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=train_cfg.get('pin_memory', True),
        drop_last=True
    )  
    # Create validation dataset and dataloader
    val_transform = get_val_transform(val_cfg) if val_cfg.get('transform') else None
    val_dataset = ImageRestorationDataset(val_cfg['input'], val_cfg['target'], transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=val_cfg.get('num_workers', 4),
        pin_memory=val_cfg.get('pin_memory', True),
        drop_last=False
    )
    logger.info(f'Dataset stats: train: {len(train_dataset)}, val: {len(val_dataset)}')
    if train_loader is None:
        logger.error("No training dataloader created. Check your configuration.")
        return
    
    # Create model from config
    model_type = model_cfg.pop('type', 'RetinexFormer')  # Remove 'type' from config
    if model_type == 'RetinexFormer':
        model = RetinexFormer(**model_cfg)  # Now model_cfg doesn't contain 'type'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = model.to(device)
    logger.info(f"Created model: {model_type} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup optimizer
    optimizer = setup_optimizer(model, train_cfg)
    
    # Setup scheduler
    scheduler = setup_scheduler(optimizer, train_cfg)
    
    # Setup loss function
    criterion = setup_loss(train_cfg)

    # Get training parameters
    total_epochs = train_cfg.get('num_epochs', 1)
    save_interval = train_cfg.get('save_interval', 1)
    val_interval = train_cfg.get('val_interval', 1)
    resume_path = train_cfg.get('resume', '~')
    
    # Initialize training state
    current_epoch = 0
    current_iter = 0
    best_val_loss = float('inf')

    # Resume training if specified
    if resume_path is not None and resume_path != '~' and os.path.exists(resume_path):
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint.get('epoch', 0)
        current_iter = checkpoint.get('iter', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {current_epoch}, iteration {current_iter}")
    elif resume_path == '~':
        logger.info("No resume specified (using ~)")
    else:
        logger.info(f"Resume path specified but file not found: {resume_path}")
    
    logger.info("Training setup complete!")
    logger.debug(f"Total epochs: {total_epochs}")
    logger.debug(f"Save interval: {save_interval}")
    logger.debug(f"Validation interval: {val_interval}")
    if resume_path is None:
        logger.info("Resume path: None (no resume)")
    else:
        logger.info(f"Resume path: {resume_path}")
    logger.info(f"Starting from iteration: {current_iter}")


    print(f"All configs and logs are saved in {log_file}")
    
    # Start training loop
    logger.info(f"Starting training from epoch {current_epoch+1}, iteration {current_iter}...")

    for epoch in range(current_epoch, total_epochs):
        # Train one epoch
        epoch_start = time.time()
        logger.debug(f"Starting training for epoch {epoch+1} (Iteration: {current_iter})")
        train_losses, current_iter = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, current_iter)

        # Update learning rate
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Training line
        logger.info(
            f"Epoch {epoch+1}/{total_epochs}: TRAIN: "
            f"Loss {train_losses['avg_total_loss']:.6f} | "
            f"L1 {train_losses['avg_l1_loss']:.6f} | "
            f"MSE {train_losses['avg_mse_loss']:.6f} | "
            f"Charb {train_losses['avg_charbonnier_loss']:.6f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"Iter {current_iter} | "
            f"Time {epoch_time:.1f}s"
        )
        
        # Validation
        if (epoch + 1) % val_interval == 0:
            val_start = time.time()
            val_losses, current_iter = validate_one_epoch(model, val_loader, criterion, device, epoch, logger, current_iter)
            val_time = time.time() - val_start
            
            # Validation line
            logger.info(
                f"Epoch {epoch+1}/{total_epochs}: VAL: "
                f"Loss {val_losses['avg_total_loss']:.6f} | "
                f"L1 {val_losses['avg_l1_loss']:.6f} | "
                f"MSE {val_losses['avg_mse_loss']:.6f} | "
                f"Charb {val_losses['avg_charbonnier_loss']:.6f} | "
                f"Iter {current_iter} | "
                f"Time {val_time:.1f}s"
            )
            
            # Save best model
            if val_losses['avg_total_loss'] < best_val_loss:
                best_val_loss = val_losses['avg_total_loss']
                logger.info(f"New best validation loss: {best_val_loss:.6f} (Iteration: {current_iter})")
                # Save best weights only (.pt)
                best_weights_path = os.path.join(log_dir, "best.pt")
                torch.save(model.state_dict(), best_weights_path)
                logger.info(f"Best weights saved: {best_weights_path}")
        
        # Save checkpoint at intervals (.pt weights only)
        if (epoch + 1) % save_interval == 0:
            logger.info(f"Starting checkpoint save for epoch {epoch+1} (Iteration: {current_iter})")
            checkpoint_path = os.path.join(log_dir, f"ckpt_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path} (Iteration: {current_iter})")
        
        # Update current epoch and log progress
        current_epoch = epoch + 1
        logger.info(f"Completed epoch {current_epoch}/{total_epochs} - Total iterations: {current_iter}")

    # Always save last weights (.pt)
    last_weights_path = os.path.join(log_dir, "last.pt")
    torch.save(model.state_dict(), last_weights_path)
    logger.info(f"Last weights saved: {last_weights_path}")
    # Ensure best.pt exists even if no validation improvement occurred
    best_weights_path = os.path.join(log_dir, "best.pt")
    if not os.path.exists(best_weights_path):
        torch.save(model.state_dict(), best_weights_path)
        logger.info(f"No best model recorded during training. Saved current model as best: {best_weights_path}")
    
    # Final training summary
    logger.info("=" * 80)
    logger.info("FINAL TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total epochs completed: {total_epochs}")
    logger.info(f"Total iterations: {current_iter}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    logger.info("=" * 80)
    
    logger.info(f"Training completed! Final iteration: {current_iter}")
    print(f"Training completed! All logs and models saved in {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train RetinexFormer model by refactoring the original code to remove the basicsr dependency")
    parser.add_argument('--cfg', type=str, required=True, help='Path to the YAML config file.')
    
    args = parser.parse_args()
    main(args)