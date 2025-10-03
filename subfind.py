#!/usr/bin/env python3
"""
Sub-image finder tool.

Takes two PNG image file paths and determines if the second image
exists within the first image using exact pixel matching.
Reports coordinates if found.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def load_image_to_gpu(image_path: Path, device: str) -> torch.Tensor:
    """Load PNG image and convert to GPU tensor as uint8 to save memory."""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None

        with Image.open(image_path) as img:
            # Load the image data to prevent lazy loading
            img.load()

            # Ensure RGB format
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy array as uint8 then to tensor and move to GPU
            img_array = np.array(img, dtype=np.uint8)
            img_tensor = torch.from_numpy(img_array).to(device)

            return img_tensor

    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        sys.exit(1)


def find_subimage_vectorized_gpu(
    large_img: torch.Tensor, small_img: torch.Tensor
) -> Optional[Tuple[int, int]]:
    """Vectorized template matching using convolution for exact pixel matching."""
    large_h, large_w = large_img.shape[:2]
    small_h, small_w = small_img.shape[:2]
    
    if small_h > large_h or small_w > large_w:
        return None
    
    # Reshape for convolution: (batch, channels, height, width)
    large_tensor = large_img.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    small_tensor = small_img.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, h, w)
    
    # Use convolution to compute correlation at all positions simultaneously
    correlation = F.conv2d(large_tensor, small_tensor, padding=0)
    
    # For exact matching, compute sum of squared differences at all positions
    large_squared = F.conv2d(large_tensor.pow(2), torch.ones_like(small_tensor), padding=0)
    small_squared = (small_tensor.pow(2)).sum()
    cross_term = 2 * correlation
    
    diff_squared = large_squared - cross_term + small_squared
    
    # Find positions where difference is essentially zero (accounting for floating point)
    matches = diff_squared < 1e-6
    
    if matches.any():
        # Get the first match
        match_indices = torch.nonzero(matches.squeeze(), as_tuple=False)
        if len(match_indices) > 0:
            y, x = match_indices[0].cpu().numpy()
            return (int(x), int(y))
    
    return None


def find_subimage_chunked_gpu(
    large_img: torch.Tensor, small_img: torch.Tensor, chunk_size: int = 2048
) -> Optional[Tuple[int, int]]:
    """Process large image in chunks to manage GPU memory."""
    large_h, large_w = large_img.shape[:2]
    small_h, small_w = small_img.shape[:2]
    
    if small_h > large_h or small_w > large_w:
        return None
    
    overlap = max(small_h, small_w) - 1
    
    for start_y in range(0, large_h - small_h + 1, chunk_size - overlap):
        for start_x in range(0, large_w - small_w + 1, chunk_size - overlap):
            end_y = min(start_y + chunk_size, large_h)
            end_x = min(start_x + chunk_size, large_w)
            
            chunk = large_img[start_y:end_y, start_x:end_x]
            
            # Use vectorized search on this chunk
            result = find_subimage_vectorized_gpu(chunk, small_img)
            if result:
                chunk_x, chunk_y = result
                return (start_x + chunk_x, start_y + chunk_y)
        
        if start_y % 1000 == 0:
            print(f"Processed rows: {start_y}/{large_h - small_h + 1}")
    
    return None


def find_subimage_exact_gpu(
    large_img: torch.Tensor, small_img: torch.Tensor
) -> Optional[Tuple[int, int]]:
    """
    Find if small image exists within large image using optimized exact pixel matching.
    Returns (x, y) coordinates of top-left corner if found, None otherwise.
    """
    # Try vectorized approach first for smaller images
    try:
        return find_subimage_vectorized_gpu(large_img, small_img)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory exceeded, falling back to chunked processing...")
            torch.cuda.empty_cache()
            return find_subimage_chunked_gpu(large_img, small_img)
        else:
            raise e


def main():
    parser = argparse.ArgumentParser(
        description="Find if second image exists within first image"
    )
    parser.add_argument("large_image", help="Path to the larger PNG image")
    parser.add_argument("small_image", help="Path to the smaller PNG image to find")

    args = parser.parse_args()

    # Validate input files
    large_path = Path(args.large_image)
    small_path = Path(args.small_image)

    if not large_path.exists():
        print(f"Error: Large image file {large_path} does not exist")
        sys.exit(1)

    if not small_path.exists():
        print(f"Error: Small image file {small_path} does not exist")
        sys.exit(1)

    # Determine device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images directly to GPU/device
    large_img = load_image_to_gpu(large_path, device)
    small_img = load_image_to_gpu(small_path, device)

    # Find subimage using exact matching on GPU
    result = find_subimage_exact_gpu(large_img, small_img)

    if result:
        x, y = result
        print(f"{x},{y}")
    else:
        print("not a sub-image")


if __name__ == "__main__":
    main()
