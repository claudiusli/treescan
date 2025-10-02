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


def find_image_in_larger_gpu(small_tensor: torch.Tensor, large_tensor: torch.Tensor, 
                            device: str) -> Optional[Tuple[int, int]]:
    """
    Find if small image exists within large image using GPU pixel-by-pixel matching.
    Returns (x, y) coordinates if found, None otherwise.
    """
    small_h, small_w = small_tensor.shape[1], small_tensor.shape[2]
    large_h, large_w = large_tensor.shape[1], large_tensor.shape[2]
    
    if small_h > large_h or small_w > large_w:
        return None
    
    # Iterate through all possible positions in the large image
    for start_y in range(large_h - small_h + 1):
        for start_x in range(large_w - small_w + 1):
            # Extract patch from large image at current position
            patch = large_tensor[:, start_y:start_y+small_h, start_x:start_x+small_w]
            
            # Compare pixel by pixel with early exit
            match = True
            for y in range(small_h):
                for x in range(small_w):
                    for c in range(3):  # RGB channels
                        if torch.abs(small_tensor[c, y, x] - patch[c, y, x]) > 0.02:  # Small tolerance for format differences
                            match = False
                            break
                    if not match:
                        break
                if not match:
                    break
            
            if match:
                return (start_x, start_y)
    
    return None


def find_image_in_larger(small_img: np.ndarray, large_img: np.ndarray, 
                        device: str = None) -> Optional[Tuple[int, int]]:
    """
    Find if small image exists within large image using GPU-accelerated pixel matching.
    Returns (x, y) coordinates if found, None otherwise.
    """
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert to PyTorch tensors and move to GPU
    small_tensor = torch.from_numpy(small_img).float().to(device) / 255.0
    large_tensor = torch.from_numpy(large_img).float().to(device) / 255.0
    
    # Rearrange dimensions from HWC to CHW
    small_tensor = small_tensor.permute(2, 0, 1)  # [C, H, W]
    large_tensor = large_tensor.permute(2, 0, 1)  # [C, H, W]
    
    if device == 'cuda':
        try:
            return find_image_in_larger_gpu(small_tensor, large_tensor, device)
        except Exception as e:
            print(f"GPU matching failed, falling back to CPU: {e}")
            # Move tensors back to CPU for fallback
            small_tensor = small_tensor.cpu()
            large_tensor = large_tensor.cpu()
    
    # CPU fallback with same logic
    small_h, small_w = small_tensor.shape[1], small_tensor.shape[2]
    large_h, large_w = large_tensor.shape[1], large_tensor.shape[2]
    
    if small_h > large_h or small_w > large_w:
        return None
    
    # Iterate through all possible positions in the large image
    for start_y in range(large_h - small_h + 1):
        for start_x in range(large_w - small_w + 1):
            # Extract patch from large image at current position
            patch = large_tensor[:, start_y:start_y+small_h, start_x:start_x+small_w]
            
            # Compare pixel by pixel with early exit
            match = True
            for y in range(small_h):
                for x in range(small_w):
                    for c in range(3):  # RGB channels
                        if torch.abs(small_tensor[c, y, x] - patch[c, y, x]) > 0.02:  # Small tolerance for format differences
                            match = False
                            break
                    if not match:
                        break
                if not match:
                    break
            
            if match:
                return (start_x, start_y)
    
    return None


def check_image_containment(image_files: List[Path]) -> None:
    """Check if smaller images are contained within the largest image."""
    print("\nChecking for image containment...")
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU acceleration (CUDA)")
    else:
        print(f"Using CPU processing")
    
    # First, find the largest image by file size
    images_info = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_array = np.array(img)
                size = img_array.shape[0] * img_array.shape[1]
                images_info.append((img_path, img_array, size))
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
    
    # Sort by size (largest first)
    images_info.sort(key=lambda x: x[2], reverse=True)
    
    if len(images_info) < 2:
        print("Need at least 2 images for comparison")
        return
    
    # Load only the largest image to GPU as reference
    largest_path, largest_array, largest_size = images_info[0]
    print(f"Loading reference image {largest_path.name} ({largest_array.shape}) to {device}")
    
    largest_tensor = torch.from_numpy(largest_array).float().to(device) / 255.0
    largest_tensor = largest_tensor.permute(2, 0, 1)  # CHW format
    
    # Test each smaller image against the largest one
    for small_path, small_array, small_size in images_info[1:]:
        print(f"Checking if {small_path.name} is in {largest_path.name}...")
        
        # Skip comparison if images have the same dimensions (likely same image, different format)
        if (small_array.shape[0] == largest_array.shape[0] and 
            small_array.shape[1] == largest_array.shape[1]):
            print(f"  MATCH FOUND: {small_path.name} found in {largest_path.name} at coordinates (0, 0) - same dimensions")
            continue
        
        # Load small image to GPU temporarily
        small_tensor = torch.from_numpy(small_array).float().to(device) / 255.0
        small_tensor = small_tensor.permute(2, 0, 1)  # CHW format
        
        # Perform pixel-by-pixel matching on GPU
        match_pos = find_image_in_larger_tensors(small_tensor, largest_tensor, device)
        if match_pos:
            x, y = match_pos
            print(f"  MATCH FOUND: {small_path.name} found in {largest_path.name} at coordinates ({x}, {y})")
        else:
            print(f"  No match found")
        
        # Free GPU memory for the small image
        del small_tensor
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Clean up the largest image tensor
    del largest_tensor
    torch.cuda.empty_cache() if device == 'cuda' else None


def find_image_in_larger_tensors(small_tensor: torch.Tensor, large_tensor: torch.Tensor, 
                                device: str) -> Optional[Tuple[int, int]]:
    """
    Find if small tensor exists within large tensor using pixel-by-pixel matching.
    Both tensors should already be on the GPU in CHW format.
    """
    small_h, small_w = small_tensor.shape[1], small_tensor.shape[2]
    large_h, large_w = large_tensor.shape[1], large_tensor.shape[2]
    
    if small_h > large_h or small_w > large_w:
        return None
    
    # Iterate through all possible positions in the large image
    for start_y in range(large_h - small_h + 1):
        for start_x in range(large_w - small_w + 1):
            # Extract patch from large image at current position
            patch = large_tensor[:, start_y:start_y+small_h, start_x:start_x+small_w]
            
            # Compare pixel by pixel with early exit
            match = True
            for y in range(small_h):
                for x in range(small_w):
                    for c in range(3):  # RGB channels
                        if torch.abs(small_tensor[c, y, x] - patch[c, y, x]) > 0.02:  # Small tolerance for format differences
                            match = False
                            break
                    if not match:
                        break
                if not match:
                    break
            
            if match:
                return (start_x, start_y)
    
    return None


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
