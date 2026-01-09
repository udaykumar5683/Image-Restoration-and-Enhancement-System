#!/usr/bin/env python3
"""
Exposure Correction Model - Main Application
Based on the CVPR 2022 NTIRE Workshop paper by Eyiokur et al.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.networks_mine import GlobalGenerator
from util.util import tensor2im
from models.models import create_model
from options.test_options import TestOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exposure_correction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExposureCorrectionModel:
    """Exposure Correction Model for enhancing image quality."""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the exposure correction model.
        
        Args:
            model_path: Path to the pretrained model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self) -> None:
        """Load the pretrained model with error handling."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Create a simple options object for model creation
            class SimpleOpt:
                def __init__(self, model_path):
                    self.model_path = model_path
                    self.nThreads = 1
                    self.serial_batches = True
                    self.no_flip = True
                    self.label_nc = 0  # No label channels - using 0 as per original test script
                    self.output_nc = 3
                    self.ngf = 64
                    self.n_downsample_global = 4
                    self.n_blocks_global = 9
                    self.norm = 'instance'
                    self.gpu_ids = []  # Force CPU usage for compatibility
                    self.is_shortcut = False
                    self.model = 'pix2pixHD'
                    self.checkpoints_dir = os.path.dirname(os.path.dirname(model_path))  # Parent directory
                    self.name = os.path.basename(os.path.dirname(model_path))  # Get folder name (100)
                    self.which_epoch = 'latest'
                    self.load_pretrain = ''
                    self.verbose = False
                    self.isTrain = False
                    self.continue_train = False
                    self.data_type = 32
                    self.fp16 = False
                    self.local_rank = 0
                    self.engine = False
                    self.onnx = False
                    self.export_onnx = ''
                    self.batchSize = 1
                    self.loadSize = 1024  # Match original test script
                    self.fineSize = 1024  # Match original test script
                    self.resize_or_crop = 'scale_width'
                    self.dataroot = ''
                    self.dir_A = ''
                    self.dir_B = ''
                    self.read_from_file = False
                    self.file_name_A = ''
                    self.file_name_B = ''
                    self.single_input_D = False
                    self.display_winsize = 512
                    self.tf_log = False
                    self.netG = 'global'
                    self.niter_fix_global = 0
                    self.spectral_normalization_D = False
                    self.dropout_D = False
                    self.instance_feat = False
                    self.label_feat = False
                    self.feat_num = 3
                    self.load_features = False
                    self.nef = 16
                    self.n_clusters = 10
                    self.l1_image_loss = False
                    self.l1_image_loss_coef = 1.0
                    self.no_ganFeat_loss = True
                    self.no_vgg_loss = True
                    self.no_lsgan = True
                    self.num_D = 2
                    self.ndf = 64
                    self.n_layers_D = 3
                    self.beta1 = 0.5
                    self.lr = 0.0002
                    self.lr_D = 0.0002
                    self.weight_decay = 0.0
                    self.pool_size = 0
                    self.no_html = False
                    self.results_dir = './results/'
                    self.aspect_ratio = 1.0
                    self.phase = 'test'
                    self.phase_test_type = 'test_all'
                    self.how_many = 50
                    self.cluster_path = 'features_clustered_010.npy'
                    self.use_encoded_image = False
                    self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            
            opt = SimpleOpt(self.model_path)
            
            # Create model using the original codebase
            from models.models import create_model
            self.model = create_model(opt)
            
            # Fix the model for CPU usage by overriding the encode_input method
            if self.device == 'cpu':
                original_encode_input = self.model.encode_input
                
                def cpu_encode_input(label_map, real_image=None, infer=False):
                    # Use CPU instead of CUDA
                    input_label = label_map.data.cpu()
                    input_label = Variable(input_label, volatile=infer) if hasattr(Variable, 'volatile') else Variable(input_label)
                    
                    # real images for training
                    if real_image is not None and self.model.opt.isTrain:
                        real_image = Variable(real_image.data.cpu())
                    
                    return input_label, real_image
                
                self.model.encode_input = cpu_encode_input
            
            # Fix the generator architecture for exposure correction (label_nc=0)
            # When label_nc=0, the model expects the input image to be passed as label
            # but the generator was created with 0 input channels. We need to modify it.
            if opt.label_nc == 0:
                # Get the original first layer
                original_first_layer = self.model.netG.model[1]
                
                # Create a new first layer with 3 input channels instead of 0
                import torch.nn as nn
                new_first_layer = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=0)
                
                # Initialize the weights (we can't copy from original since it has 0 channels)
                nn.init.normal_(new_first_layer.weight, 0.0, 0.02)
                if new_first_layer.bias is not None:
                    nn.init.constant_(new_first_layer.bias, 0)
                
                # Replace the first layer
                self.model.netG.model[1] = new_first_layer
                
                logger.info("Fixed generator architecture for exposure correction (modified first layer to accept 3 input channels)")
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for model input following the original approach.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image using PIL (similar to original data loader)
            img = Image.open(image_path).convert('RGB')
            
            # Resize to loadSize (1024) maintaining aspect ratio
            loadSize = 1024
            w, h = img.size
            if w < h:
                new_w = loadSize
                new_h = int(h * loadSize / w)
            else:
                new_h = loadSize
                new_w = int(w * loadSize / h)
            
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Center crop to fineSize (1024 for exposure correction)
            fineSize = 1024
            w, h = img.size
            left = (w - fineSize) // 2
            top = (h - fineSize) // 2
            right = left + fineSize
            bottom = top + fineSize
            img = img.crop((left, top, right, bottom))
            
            # Convert to numpy array and normalize to [0, 1]
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Convert to tensor and rearrange dimensions (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            
            # Normalize to [-1, 1] range (as expected by the model)
            img_tensor = (img_tensor - 0.5) / 0.5
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            
            return img_tensor
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {str(e)}")
            raise
    
    def postprocess_image(self, output_tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to displayable image following the original approach.
        
        Args:
            output_tensor: Model output tensor
            
        Returns:
            Postprocessed image array
        """
        try:
            # Use the original tensor2im utility function
            img_np = tensor2im(output_tensor)
            
            # Ensure values are in valid range
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            return img_np
            
        except Exception as e:
            logger.error(f"Failed to postprocess output using tensor2im: {str(e)}")
            # Fallback to manual conversion
            try:
                # Manual tensor to image conversion
                if output_tensor.dim() == 4:
                    output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
                
                # Convert from [-1, 1] to [0, 255]
                img_np = ((output_tensor + 1) * 127.5).clamp(0, 255).cpu().numpy()
                
                # Rearrange dimensions if needed (C, H, W) -> (H, W, C)
                if img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                return img_np.astype(np.uint8)
                
            except Exception as e2:
                logger.error(f"Alternative postprocessing also failed: {str(e2)}")
                raise
    
    def enhance_image(self, image_path: str, save_path: Optional[str] = None) -> np.ndarray:
        """
        Enhance a single image using the exposure correction model.
        
        Args:
            image_path: Path to the input image
            save_path: Optional path to save the enhanced image
            
        Returns:
            Enhanced image array
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image not found: {image_path}")
            
            # Preprocess - this creates the label tensor (which is the input image when label_nc=0)
            label_tensor = self.preprocess_image(image_path)
            label_tensor = label_tensor.to(self.device)
            
            # Run inference using the model's inference method
            with torch.no_grad():
                if hasattr(self.model, 'inference'):
                    # Use the proper inference method - pass the preprocessed image as label
                    # and None for the target image (since we're doing inference)
                    output_tensor = self.model.inference(label_tensor, None)
                else:
                    # Fallback - this shouldn't happen with the correct model
                    logger.warning("Model doesn't have inference method, using direct forward pass")
                    output_tensor = self.model.netG(label_tensor)
            
            # Postprocess
            enhanced_image = self.postprocess_image(output_tensor)
            
            # Save if path provided
            if save_path:
                self.save_image(enhanced_image, save_path)
                logger.info(f"Enhanced image saved to: {save_path}")
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Failed to enhance image {image_path}: {str(e)}")
            raise
    
    def save_image(self, image_array: np.ndarray, save_path: str) -> None:
        """
        Save image array to file.
        
        Args:
            image_array: Image array to save
            save_path: Path to save the image
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save image
            img = Image.fromarray(image_array)
            img.save(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save image {save_path}: {str(e)}")
            raise
    
    def process_batch(self, model_instance, input_dir: str, output_dir: str, image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']) -> List[str]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save enhanced images
            image_extensions: List of valid image extensions
            
        Returns:
            List of processed image paths
        """
        processed_images = []
        
        try:
            # Get all image files
            image_files = []
            for ext in image_extensions:
                image_files.extend([f for f in Path(input_dir).glob(f'*{ext}')])
                image_files.extend([f for f in Path(input_dir).glob(f'*{ext.upper()}')])
            
            if not image_files:
                logger.warning(f"No images found in {input_dir}")
                return processed_images
            
            logger.info(f"Found {len(image_files)} images to process")
            
            # Process each image
            for image_path in tqdm(image_files, desc="Processing images"):
                try:
                    # Create output path
                    output_path = os.path.join(output_dir, f"enhanced_{image_path.name}")
                    
                    # Enhance image
                    enhanced_image = model_instance.enhance_image(str(image_path), output_path)
                    processed_images.append(output_path)
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(processed_images)} images")
            return processed_images
            
        except Exception as e:
            logger.error(f"Failed to process batch: {str(e)}")
            raise

def create_comparison_plot(original_path: str, enhanced_image: np.ndarray, save_path: str) -> None:
    """Create a side-by-side comparison plot."""
    try:
        # Load original image
        original_img = Image.open(original_path)
        original_array = np.array(original_img)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        ax1.imshow(original_array)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Enhanced image
        ax2.imshow(enhanced_image)
        ax2.set_title('Enhanced Image')
        ax2.axis('off')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to create comparison plot: {str(e)}")

def main():
    """Main function to run exposure correction."""
    parser = argparse.ArgumentParser(description='Exposure Correction Model')
    parser.add_argument('--input', type=str, default='test', help='Input image or directory path')
    parser.add_argument('--output', type=str, default='results', help='Output directory path')
    parser.add_argument('--model', type=str, default='checkpoints/100/latest_net_G.pth', help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--create-comparison', action='store_true', help='Create comparison plots')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Determine device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        logger.info(f"Using device: {device}")
        
        # Initialize model
        logger.info("Initializing exposure correction model...")
        model = ExposureCorrectionModel(args.model, device)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Process input
        if os.path.isfile(args.input):
            # Single image
            logger.info(f"Processing single image: {args.input}")
            output_path = os.path.join(args.output, f"enhanced_{os.path.basename(args.input)}")
            enhanced_image = model.enhance_image(args.input, output_path)
            
            # Create comparison plot if requested
            if args.create_comparison:
                comparison_path = os.path.join(args.output, f"comparison_{os.path.basename(args.input)}.png")
                create_comparison_plot(args.input, enhanced_image, comparison_path)
                
        elif os.path.isdir(args.input):
            # Directory of images
            logger.info(f"Processing directory: {args.input}")
            processed_images = model.process_batch(model, args.input, args.output)
            
            # Create comparison plots if requested
            if args.create_comparison and processed_images:
                logger.info("Creating comparison plots...")
                for processed_path in processed_images:
                    # Find original image
                    original_name = os.path.basename(processed_path).replace('enhanced_', '')
                    original_path = os.path.join(args.input, original_name)
                    
                    if os.path.exists(original_path):
                        enhanced_img = cv2.imread(processed_path)
                        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                        comparison_path = os.path.join(args.output, f"comparison_{original_name}.png")
                        create_comparison_plot(original_path, enhanced_img, comparison_path)
        else:
            logger.error(f"Input path does not exist: {args.input}")
            return 1
        
        logger.info("Exposure correction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Exposure correction failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())