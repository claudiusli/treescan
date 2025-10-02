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
from typing import Optional, Tuple


def load_image_as_array(image_path: Path) -> np.ndarray:
    """Load PNG image and convert to numpy array."""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None
        
        with Image.open(image_path) as img:
            # Load the image data to prevent lazy loading
            img.load()
            
            # Ensure RGB format
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            return np.array(img)
            
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        sys.exit(1)


def find_subimage_exact(large_img: np.ndarray, small_img: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Find if small image exists within large image using exact pixel matching.
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
            # Compare pixel by pixel with early exit on mismatch
            match = True
            for y in range(small_h):
                for x in range(small_w):
                    for c in range(3):  # RGB channels
                        if large_img[start_y + y, start_x + x, c] != small_img[y, x, c]:
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
    parser = argparse.ArgumentParser(description="Find if second image exists within first image")
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
    
    # Load images
    large_img = load_image_as_array(large_path)
    small_img = load_image_as_array(small_path)
    
    # Find subimage using exact matching
    result = find_subimage_exact(large_img, small_img)
    
    if result:
        x, y = result
        print(f"{x},{y}")
    else:
        print("not a sub-image")


if __name__ == "__main__":
    main()
