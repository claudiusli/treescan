#!/usr/bin/env python3
"""
Image format normalization and matching tool.

Takes a directory path, creates a new directory with "_fix" suffix,
converts all images to a consistent format, and checks if smaller
images are contained within larger images.
"""

import sys
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}
    image_files = []
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def normalize_image_format(image_path: Path, output_path: Path, target_format: str = 'PNG') -> None:
    """Convert image to target format with consistent properties."""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None
        
        with Image.open(image_path) as img:
            # Load the image data to prevent lazy loading
            img.load()
            
            # Always convert to RGB to ensure consistent channel ordering
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save in target format
            output_file = output_path / f"{image_path.stem}.{target_format.lower()}"
            img.save(output_file, format=target_format)
            print(f"Converted {image_path.name} -> {output_file.name}")
            
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")


def find_image_in_larger_gpu(small_img: np.ndarray, large_img: np.ndarray, 
                            threshold: float = 0.99, device: str = None) -> Optional[Tuple[int, int]]:
    """
    Find if small image exists within large image using GPU-accelerated template matching.
    Returns (x, y) coordinates if found, None otherwise.
    """
    if small_img.shape[0] > large_img.shape[0] or small_img.shape[1] > large_img.shape[1]:
        return None
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert to grayscale for faster matching
    if len(small_img.shape) == 3:
        small_gray = np.mean(small_img, axis=2)
        large_gray = np.mean(large_img, axis=2)
    else:
        small_gray = small_img
        large_gray = large_img
    
    # Convert to PyTorch tensors and move to GPU
    small_tensor = torch.from_numpy(small_gray).float().to(device) / 255.0
    large_tensor = torch.from_numpy(large_gray).float().to(device) / 255.0
    
    # Add batch and channel dimensions for conv2d
    small_tensor = small_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    large_tensor = large_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Use normalized cross-correlation via convolution
    # Flip the small image for convolution (correlation = convolution with flipped kernel)
    small_flipped = torch.flip(small_tensor, [2, 3])
    
    # Perform template matching using F.conv2d
    correlation_map = F.conv2d(large_tensor, small_flipped, padding=0)
    
    # Normalize the correlation map
    h, w = small_tensor.shape[2], small_tensor.shape[3]
    
    # Calculate local means and standard deviations for normalization
    ones_kernel = torch.ones(1, 1, h, w, device=device)
    large_mean = F.conv2d(large_tensor, ones_kernel, padding=0) / (h * w)
    large_sq_mean = F.conv2d(large_tensor ** 2, ones_kernel, padding=0) / (h * w)
    large_std = torch.sqrt(large_sq_mean - large_mean ** 2)
    
    small_mean = torch.mean(small_tensor)
    small_std = torch.std(small_tensor)
    
    # Normalize correlation
    normalized_corr = (correlation_map / (h * w) - large_mean * small_mean) / (large_std * small_std + 1e-8)
    
    # Find the best match
    max_corr, max_idx = torch.max(normalized_corr.flatten(), 0)
    
    if max_corr.item() > threshold:
        # Convert flat index back to 2D coordinates
        corr_h, corr_w = normalized_corr.shape[2], normalized_corr.shape[3]
        y = max_idx.item() // corr_w
        x = max_idx.item() % corr_w
        return (x, y)
    
    return None


def find_image_in_larger_early_exit(small_img: np.ndarray, large_img: np.ndarray, 
                                   threshold: float = 0.99, tolerance: float = 0.01) -> Optional[Tuple[int, int]]:
    """
    CPU fallback with early exit optimization for exact pixel matching.
    Returns (x, y) coordinates if found, None otherwise.
    """
    if small_img.shape[0] > large_img.shape[0] or small_img.shape[1] > large_img.shape[1]:
        return None
    
    h, w = small_img.shape[:2]
    large_h, large_w = large_img.shape[:2]
    
    # For exact matching, compare pixel by pixel with early exit
    for y in range(large_h - h + 1):
        for x in range(large_w - w + 1):
            # Extract patch from large image
            patch = large_img[y:y+h, x:x+w]
            
            # Early exit comparison - stop as soon as we find a mismatch
            match = True
            if len(small_img.shape) == 3:  # Color image
                for i in range(h):
                    for j in range(w):
                        if np.any(np.abs(small_img[i, j] - patch[i, j]) > tolerance * 255):
                            match = False
                            break
                    if not match:
                        break
            else:  # Grayscale
                for i in range(h):
                    for j in range(w):
                        if abs(small_img[i, j] - patch[i, j]) > tolerance * 255:
                            match = False
                            break
                    if not match:
                        break
            
            if match:
                return (x, y)
    
    return None


def find_image_in_larger(small_img: np.ndarray, large_img: np.ndarray, 
                        threshold: float = 0.99) -> Optional[Tuple[int, int]]:
    """
    Find if small image exists within large image using optimized template matching.
    Returns (x, y) coordinates if found, None otherwise.
    """
    # Try GPU acceleration first if available
    if torch.cuda.is_available():
        try:
            return find_image_in_larger_gpu(small_img, large_img, threshold)
        except Exception as e:
            print(f"GPU matching failed, falling back to CPU: {e}")
    
    # Fallback to CPU with early exit for exact matches
    if threshold >= 0.99:
        return find_image_in_larger_early_exit(small_img, large_img, threshold)
    
    # Original CPU implementation for lower thresholds
    if small_img.shape[0] > large_img.shape[0] or small_img.shape[1] > large_img.shape[1]:
        return None
    
    # Convert to grayscale for faster matching
    if len(small_img.shape) == 3:
        small_gray = np.mean(small_img, axis=2)
        large_gray = np.mean(large_img, axis=2)
    else:
        small_gray = small_img
        large_gray = large_img
    
    # Normalize to 0-1 range
    small_gray = small_gray.astype(np.float32) / 255.0
    large_gray = large_gray.astype(np.float32) / 255.0
    
    # Template matching using normalized cross-correlation
    h, w = small_gray.shape
    best_match = -1
    best_pos = None
    
    for y in range(large_gray.shape[0] - h + 1):
        for x in range(large_gray.shape[1] - w + 1):
            # Extract patch from large image
            patch = large_gray[y:y+h, x:x+w]
            
            # Calculate normalized cross-correlation
            correlation = np.corrcoef(small_gray.flatten(), patch.flatten())[0, 1]
            
            if not np.isnan(correlation) and correlation > best_match:
                best_match = correlation
                best_pos = (x, y)
    
    if best_match > threshold:
        return best_pos
    
    return None


def check_image_containment(image_files: List[Path]) -> None:
    """Check if smaller images are contained within larger images."""
    print("\nChecking for image containment...")
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU acceleration (CUDA)")
    else:
        print(f"Using CPU processing")
    
    # Load all images and sort by size
    images_data = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_array = np.array(img)
                size = img_array.shape[0] * img_array.shape[1]
                images_data.append((img_path, img_array, size))
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
    
    # Sort by size (largest first)
    images_data.sort(key=lambda x: x[2], reverse=True)
    
    # Check each smaller image against larger ones
    for i, (small_path, small_img, small_size) in enumerate(images_data[1:], 1):
        for j, (large_path, large_img, large_size) in enumerate(images_data[:i]):
            print(f"Checking if {small_path.name} is in {large_path.name}...")
            
            match_pos = find_image_in_larger(small_img, large_img)
            if match_pos:
                x, y = match_pos
                print(f"  MATCH FOUND: {small_path.name} found in {large_path.name} at coordinates ({x}, {y})")
            else:
                print(f"  No match found")


def main():
    parser = argparse.ArgumentParser(description="Normalize image formats and check for containment")
    parser.add_argument("directory", help="Source directory containing images")
    parser.add_argument("--format", default="PNG", choices=["PNG", "JPEG", "TIFF"], 
                       help="Target image format (default: PNG)")
    
    args = parser.parse_args()
    
    # Validate source directory
    source_dir = Path(args.directory)
    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist")
        sys.exit(1)
    
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        sys.exit(1)
    
    # Create target directory
    target_dir = source_dir.parent / f"{source_dir.name}_fix"
    target_dir.mkdir(exist_ok=True)
    print(f"Created target directory: {target_dir}")
    
    # Get all image files
    image_files = get_image_files(source_dir)
    if not image_files:
        print(f"No image files found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files")
    
    # Normalize all images to target format
    print(f"\nNormalizing images to {args.format} format...")
    for img_path in image_files:
        normalize_image_format(img_path, target_dir, args.format)
    
    # Check for image containment using original files
    check_image_containment(image_files)
    
    print(f"\nProcessing complete. Normalized images saved to: {target_dir}")


if __name__ == "__main__":
    main()
