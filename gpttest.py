import numpy as np
import sys
from pathlib import Path
from PIL import Image


def rabin_karp_2d_search(big, small):
    """
    big:  H x W x C uint8
    small:h x w x C uint8
    Returns (y, x) top-left if found, else None.
    Deterministic, exact (hash + memcmp verify).
    """
    H, W, C = big.shape
    h, w, c = small.shape
    assert c == C, "Channel mismatch after normalization"

    if h > H or w > W:
        return None

    # Parameters (two independent hashes)
    B1, M1 = np.uint64(1315423911), np.uint64(
        0x1FFFFFFFFFFFFFFF
    )  # large odd mask (used only if you prefer &M)
    B2, M2 = np.uint64(2654435761), np.uint64(0x1FFFFFFFFFFFFFFF)

    # Precompute powers for width and height for both hashes
    pow_w1 = np.power(B1, np.arange(w, dtype=np.uint64), dtype=np.uint64)
    pow_w2 = np.power(B2, np.arange(w, dtype=np.uint64), dtype=np.uint64)
    pow_h1 = np.power(B1, np.arange(h, dtype=np.uint64), dtype=np.uint64)
    pow_h2 = np.power(B2, np.arange(h, dtype=np.uint64), dtype=np.uint64)


    # Prepack rows for big and small (pack channels into a 32-bit integer per pixel)
    def pack_channels(img, C):
        # img: (H, W, C) uint8
        packed = np.zeros(img.shape[:2], dtype=np.uint32)
        for i in range(C):
            packed |= img[:, :, i].astype(np.uint32) << (8 * i)
        return packed.astype(np.uint64)

    big_pack = pack_channels(big, C)  # (H, W) uint64
    small_pack = pack_channels(small, C)  # (h, w) uint64

    # Rolling hash across width for every row
    def rolling_width(arr, pow_w, B):
        # arr: (H, W) uint64 → out: (H, W-w+1) uint64
        Hh, Ww = arr.shape
        out = np.empty((Hh, Ww - w + 1), dtype=np.uint64)
        # initial hash per row
        init = (arr[:, :w] * pow_w[::-1]).sum(axis=1, dtype=np.uint64)
        out[:, 0] = init
        # slide
        head_weight = pow_w[-1]
        for x in range(1, Ww - w + 1):
            out[:, x] = (out[:, x - 1] - arr[:, x - 1] * head_weight) * B + arr[
                :, x + w - 1
            ]
        return out

    rowhash1 = rolling_width(big_pack, pow_w1, B1)  # (H, W-w+1)
    rowhash2 = rolling_width(big_pack, pow_w2, B2)

    # Row hashes for small
    small_row1 = (small_pack * pow_w1[::-1]).sum(axis=1, dtype=np.uint64)
    small_row2 = (small_pack * pow_w2[::-1]).sum(axis=1, dtype=np.uint64)

    # Now roll vertically over height on the row-hash images
    def rolling_height(rowhash, pow_h, B):
        Hh, Sw = rowhash.shape
        out = np.empty((Hh - h + 1, Sw), dtype=np.uint64)
        init = (rowhash[:h, :] * pow_h[::-1, None]).sum(axis=0, dtype=np.uint64)
        out[0, :] = init
        head_weight = pow_h[-1]
        for y in range(1, Hh - h + 1):
            out[y, :] = (
                out[y - 1, :] - rowhash[y - 1, :] * head_weight
            ) * B + rowhash[y + h - 1, :]
        return out

    big_hash1 = rolling_height(rowhash1, pow_h1, B1)  # (H-h+1, W-w+1)
    big_hash2 = rolling_height(rowhash2, pow_h2, B2)

    small_hash1 = (small_row1 * pow_h1[::-1]).sum(dtype=np.uint64)
    small_hash2 = (small_row2 * pow_h2[::-1]).sum(dtype=np.uint64)

    # Candidate positions where both hashes match
    candidates = np.where((big_hash1 == small_hash1) & (big_hash2 == small_hash2))
    for y, x in zip(*candidates):
        # Verify exact equality to eliminate any collision:
        if np.array_equal(big[y : y + h, x : x + w, :], small):
            return (y, x)

    return None


def load_image(image_path):
    """Load image and convert to uint8 RGB format."""
    # Disable decompression bomb protection for large images
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(image_path) as img:
        # Load the image data to prevent lazy loading
        img.load()

        # Ensure RGB format
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to numpy array
        img_array = np.array(img, dtype=np.uint8)
        return img_array


def main():
    if len(sys.argv) != 3:
        print("Usage: python gpttest.py <large_image> <small_image>")
        sys.exit(1)

    large_path = Path(sys.argv[1])
    small_path = Path(sys.argv[2])

    if not large_path.exists():
        print(f"Error: Large image file {large_path} does not exist")
        sys.exit(1)

    if not small_path.exists():
        print(f"Error: Small image file {small_path} does not exist")
        sys.exit(1)

    # Load images
    print("Loading large image...")
    big = load_image(large_path)
    print(f"Large image shape: {big.shape}")

    print("Loading small image...")
    small = load_image(small_path)
    print(f"Small image shape: {small.shape}")

    # Search for subimage
    print("Searching for subimage...")
    result = rabin_karp_2d_search(big, small)

    if result:
        y, x = result
        print(f"{x},{y}")
    else:
        print("not a sub-image")


if __name__ == "__main__":
    main()
