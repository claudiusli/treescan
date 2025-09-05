#!/usr/bin/env python3
"""
test_peek_pixels.py
Test script that steps across an image in N x N patches
and calls get_patch from peek_pixels.
"""

import sys
from peek_pixels import get_patch, load_first_frame


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_peek_pixels.py <image_file> <N>")
        sys.exit(1)

    fpath = sys.argv[2]
    n = int(sys.argv[1])

    # Get the image dimensions
    im = load_first_frame(fpath)
    width, height = im.size

    x, y = 0, 0
    while y + n <= height:
        while x + n <= width:
            patch = get_patch(fpath, x, y, n)
            # Not doing anything with the patch yet
            print(f"Patch at (x={x}, y={y})")
            x += 1
        x = 0
        y += 1


if __name__ == "__main__":
    main()
