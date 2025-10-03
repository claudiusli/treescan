import numpy as np

def to_uint8_rgb(arr):
    """
    Expect arr as HxWxC from your decoder (e.g., Pillow/OpenCV).
    Convert to uint8 RGB deterministically.
    """
    # If grayscale -> stack to RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        # Keep RGBA or drop alpha depending on your use:
        arr = arr[:, :, :3]  # drop alpha for canonical RGB
    if arr.dtype != np.uint8:
        # If 16-bit or float, map deterministically to 8-bit (here: simple right shift / clip)
        if np.issubdtype(arr.dtype, np.integer):
            # e.g., 16-bit to 8-bit
            bits = arr.dtype.itemsize * 8
            if bits > 8:
                arr = (arr >> (bits - 8)).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        else:
            # float in [0,1] -> [0,255]
            arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)

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
    B1, M1 = np.uint64(1315423911), np.uint64(0x1fffffffffffffff)  # large odd mask (used only if you prefer &M)
    B2, M2 = np.uint64(2654435761), np.uint64(0x1fffffffffffffff)

    # Precompute powers for width and height for both hashes
    pow_w1 = np.power(B1, np.arange(w, dtype=np.uint64), dtype=np.uint64)
    pow_w2 = np.power(B2, np.arange(w, dtype=np.uint64), dtype=np.uint64)
    pow_h1 = np.power(B1, np.arange(h, dtype=np.uint64), dtype=np.uint64)
    pow_h2 = np.power(B2, np.arange(h, dtype=np.uint64), dtype=np.uint64)

    # Helper: hash one row window of length w (over channels)
    def hash_row_segment(row):  # row is shape (W, C)
        # Flatten channel bytes into a base-256 integer stream by mixing bytes directly
        # Efficiently combine channels: dot with powers over columns; channels folded by base-257 step
        # Simpler: treat per-pixel as a 32-bit integer (for C<=4) to reduce ops deterministically
        if C <= 4:
            packed = np.zeros((row.shape[0],), dtype=np.uint32)
            for i in range(C):
                packed |= (row[:, i].astype(np.uint32) << (8 * i))
            return packed.astype(np.uint64)
        else:
            # For many channels, reduce deterministically
            return (row.astype(np.uint64) * np.uint64(257) ** np.arange(C, dtype=np.uint64)).sum(axis=1, dtype=np.uint64)

    # Prepack rows for big and small
    big_pack = np.apply_along_axis(hash_row_segment, 2, big.transpose(0,2,1)).transpose(0,2,1)  # (H, W)
    small_pack = hash_row_segment(small.reshape(-1, C)).reshape(h, w)

    # Rolling hash across width for every row
    def rolling_width(arr, pow_w):
        # arr: (H, W) uint64 → out: (H, W-w+1) uint64
        Hh, Ww = arr.shape
        out = np.empty((Hh, Ww - w + 1), dtype=np.uint64)
        # initial hash per row
        init = (arr[:, :w] * pow_w[::-1]).sum(axis=1, dtype=np.uint64)
        out[:, 0] = init
        # slide
        head_weight = pow_w[-1]
        for x in range(1, Ww - w + 1):
            out[:, x] = (out[:, x-1] - arr[:, x-1] * head_weight) * B1 + arr[:, x + w - 1]
        return out

    rowhash1 = rolling_width(big_pack, pow_w1)  # (H, W-w+1)
    rowhash2 = rolling_width(big_pack, pow_w2)

    # Row hashes for small
    small_row1 = (small_pack * pow_w1[::-1]).sum(axis=1, dtype=np.uint64)
    small_row2 = (small_pack * pow_w2[::-1]).sum(axis=1, dtype=np.uint64)

    # Now roll vertically over height on the row-hash images
    def rolling_height(rowhash, pow_h):
        Hh, Sw = rowhash.shape
        out = np.empty((Hh - h + 1, Sw), dtype=np.uint64)
        init = (rowhash[:h, :] * pow_h[::-1, None]).sum(axis=0, dtype=np.uint64)
        out[0, :] = init
        head_weight = pow_h[-1]
        for y in range(1, Hh - h + 1):
            out[y, :] = (out[y-1, :] - rowhash[y-1, :] * head_weight) * B1 + rowhash[y + h - 1, :]
        return out

    big_hash1 = rolling_height(rowhash1, pow_h1)  # (H-h+1, W-w+1)
    big_hash2 = rolling_height(rowhash2, pow_h2)

    small_hash1 = (small_row1 * pow_h1[::-1]).sum(dtype=np.uint64)
    small_hash2 = (small_row2 * pow_h2[::-1]).sum(dtype=np.uint64)

    # Candidate positions where both hashes match
    candidates = np.where((big_hash1 == small_hash1) & (big_hash2 == small_hash2))
    for y, x in zip(*candidates):
        # Verify exact equality to eliminate any collision:
        if np.array_equal(big[y:y+h, x:x+w, :], small):
            return (y, x)

    return None
