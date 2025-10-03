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
from typing import Optional, Tuple


def load_image_to_gpu(image_path: Path, device: str) -> torch.Tensor:
    """Load PNG image and convert to GPU tensor."""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None

        with Image.open(image_path) as img:
            # Load the image data to prevent lazy loading
            img.load()

            # Ensure RGB format
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy array then to tensor and move to GPU
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).to(device)

            return img_tensor

    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        sys.exit(1)


def find_subimage_exact_gpu(
    large_img: torch.Tensor, small_img: torch.Tensor
) -> Optional[Tuple[int, int]]:
    """
    Find if small image exists within large image using exact pixel matching on GPU.
    Returns (x, y) coordinates of top-left corner if found, None otherwise.
    """
    large_h, large_w = large_img.shape[:2]
    small_h, small_w = small_img.shape[:2]

    # Check if small image can fit in large image
    if small_h > large_h or small_w > large_w:
        return None

    # Iterate through all possible positions in the large image
    for start_y in range(large_h - small_h + 1):
        for start_x in range(large_w - small_w + 1):
            # Extract patch from large image at current position
            patch = large_img[start_y : start_y + small_h, start_x : start_x + small_w]

            # Compare using tensor equality with early exit
            if torch.equal(small_img, patch):
                return (start_x, start_y)
        print(f"x: {start_x}, y: {start_y}")
    return None


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
