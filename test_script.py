import argparse
import sys
from peek_pixels import get_patch
from simpleshow import show_image

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display an N x N pixel block at (X, Y) from an image"
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
        # Get the patch from peek_pixels
        patch = get_patch(fpath, x, y, n)
        # Display using simpleshow
        show_image(patch)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
