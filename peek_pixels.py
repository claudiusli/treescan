"""
peek_pixels.py
Read an image (TIFF/JPEG/PNG/...) and print its dimensions.
Also print and save the N x N pixel block whose top-left corner is at (X, Y).
Usage:
    python peek_pixels.py X Y N F
Example:
    python peek_pixels.py 10 20 5 /path/to/image.tif
"""

import argparse
import os
import sys
from typing import Tuple
from PIL import Image, ImageSequence
import numpy as np


def load_first_frame(path: str) -> Image.Image:
    """Load the first frame of an image (handles multi-frame TIFFs)."""
    im = Image.open(path)
    try:
        # For multi-frame formats like TIFF, select the first frame
        if getattr(im, "is_animated", False):
            im.seek(0)
        else:
            # Some TIFFs expose frames via ImageSequence
            for i, frame in enumerate(ImageSequence.Iterator(im)):
                # Take the first and break
                im = frame.copy()
                break
    except Exception:
        # If anything odd happens, just return the opened image
        pass
    return im


def validate_box(x: int, y: int, n: int, size: Tuple[int, int]) -> None:
    w, h = size
    if x < 0 or y < 0:
        raise ValueError(f"X and Y must be non-negative (got X={x}, Y={y}).")
    if n <= 0:
        raise ValueError(f"N must be a positive integer (got N={n}).")
    if x + n > w or y + n > h:
        raise ValueError(
            f"Requested {n}x{n} block at (X={x}, Y={y}) exceeds image bounds "
            f"{w}x{h}. Max allowed N at this position is "
            f"{min(w - x, h - y)}."
        )


def get_patch(fpath: str, x: int, y: int, n: int) -> Image.Image:
    """Get an N x N pixel patch from the image at (x, y)"""
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File not found: {fpath}")
    
    # Load image (first frame if multi-frame TIFF)
    im = load_first_frame(fpath)
    w, h = im.size
    
    # Validate and crop
    validate_box(x, y, n, (w, h))
    
    box = (x, y, x + n, y + n)
    patch = im.crop(box)
    
    # Ensure patch is in a standard format
    if patch.mode not in ("RGB", "RGBA", "L"):
        patch = patch.convert("RGB")
    return patch

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print image size and show an N x N pixel block at (X, Y)."
    )
    parser.add_argument("X", type=int, help="Top-left X coordinate (0-based)")
    parser.add_argument("Y", type=int, help="Top-left Y coordinate (0-based)")
    parser.add_argument("N", type=int, help="Size of the square block (N x N)")
    parser.add_argument(
        "F", type=str, help="Path to the image file (TIFF/JPG/PNG, etc.)"
    )
    args = parser.parse_args()

    x, y, n, fpath = args.X, args.Y, args.N, args.F

    try:
        patch = get_patch(fpath, x, y, n)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load original image to print info
    im = load_first_frame(fpath)
    w, h = im.size
    print(f"Image: {fpath}")
    print(f"Mode: {im.mode}")
    print(f"Dimensions (WxH): {w} x {h} pixels")

    # Convert to a NumPy array for readable printing
    arr = np.array(patch)
    print(f"\nPatch top-left: (X={x}, Y={y}), size: {n} x {n}")
    print(
        f"Patch array shape: {arr.shape} (H x W x C for color images; H x W for grayscale)"
    )
    print("Pixel values:")
    # Print the full array (for large N this can be long). Adjust print options if desired.
    with np.printoptions(edgeitems=10, linewidth=120, threshold=10000):
        print(arr)

    # Save the patch next to the source image for convenience
    base = os.path.basename(fpath)
    root, _ = os.path.splitext(base)
    out_name = f"{root}_patch_x{x}_y{y}_n{n}.png"
    out_path = os.path.join(os.getcwd(), out_name)
    patch.save(out_path)
    print(f"\nSaved patch image to: {out_path}")


if __name__ == "__main__":
    main()
