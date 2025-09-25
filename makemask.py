import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import argparse
import os


# Define the same network architecture used in classifier.py
class StainDiscriminator(nn.Module):
    def __init__(self, window_size):
        super(StainDiscriminator, self).__init__()

        # Focus on color channel relationships - use 1x1 convolutions to learn color combinations
        self.color_attention = nn.Sequential(
            nn.Conv2d(3, 16, 1),  # 1x1 conv to learn color relationships
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),  # Project back to 3 channels
            nn.Sigmoid(),  # Attention weights for color channels
        )

        # Simple feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Global average pooling for spatial invariance
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, x):
        # Apply color attention to emphasize stain differences
        color_weights = self.color_attention(x)
        x = x * color_weights  # Weight color channels

        # Extract features
        features = self.feature_extractor(x)

        # Global pooling and classification
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        return self.classifier(flattened)


def calculate_max_batch_size(window_size, device, net):
    """Calculate maximum batch size that fits in VRAM"""
    if device.type == 'cpu':
        return 1024  # Use a reasonable batch size for CPU
    
    # Get available VRAM (leave 20% buffer for overhead)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory
    usable_memory = free_memory * 0.8  # 20% buffer
    
    # Test with a small batch to measure actual memory usage
    test_batch_size = 1
    test_input = torch.randn(test_batch_size, 3, window_size, window_size, device=device)
    
    # Clear cache and measure memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Do a forward pass to measure actual memory usage
    with torch.no_grad():
        _ = net(test_input)
    
    # Get the peak memory used during forward pass
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Memory per patch is peak memory divided by test batch size
    memory_per_patch = peak_memory / test_batch_size
    
    # Calculate safe batch size
    max_batch_size = int(usable_memory // memory_per_patch)
    
    # Apply reasonable limits
    max_batch_size = min(max_batch_size, 4096)  # Upper limit
    max_batch_size = max(max_batch_size, 1)     # Lower limit
    
    print(f"Memory per patch: {memory_per_patch / 1024**2:.2f} MB")
    print(f"Max batch size: {max_batch_size} patches")
    return max_batch_size

def smart_batched_sliding_window(tensor, window_size, device, net):
    num_y = tensor.size(1) - window_size + 1
    num_x = tensor.size(2) - window_size + 1
    
    batch_size = calculate_max_batch_size(window_size, device, net)
    batch = []
    positions = []
    
    for y in range(num_y):
        for x in range(num_x):
            patch = tensor[:, y:y+window_size, x:x+window_size].unsqueeze(0)
            batch.append(patch)
            positions.append((y, x))
            
            # Flush batch when full, regardless of row boundaries
            if len(batch) == batch_size:
                yield torch.cat(batch, dim=0), positions
                batch = []
                positions = []
    
    if batch:  # Final partial batch
        yield torch.cat(batch, dim=0), positions

def main():
    parser = argparse.ArgumentParser(
        description="Create a mask from a JPEG image using a trained classifier"
    )
    parser.add_argument("jpeg_file", type=str, help="Path to the JPEG image file")
    parser.add_argument(
        "model_file", type=str, help="Path to the trained model weights file"
    )
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.jpeg_file):
        raise FileNotFoundError(f"JPEG file not found: {args.jpeg_file}")
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Model file not found: {args.model_file}")

    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model data
    save_data = torch.load(args.model_file, map_location=device)
    window_size = save_data["window_size"]
    print(f"Window size: {window_size}")

    # Initialize the network
    net = StainDiscriminator(window_size)
    net.load_state_dict(save_data["model_state_dict"])
    net.to(device)
    net.eval()  # Set to evaluation mode

    # Increase the maximum image size to handle large images
    Image.MAX_IMAGE_PIXELS = None

    # Load image
    with Image.open(args.jpeg_file) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Convert to tensor
        img_array = np.array(img)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).to(device)
    
    # Get dimensions
    _, height, width = img_tensor.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Calculate mask dimensions (smaller than original image)
    mask_height = height - window_size + 1
    mask_width = width - window_size + 1
    # Create mask tensor on GPU with the correct dimensions
    mask_tensor = torch.zeros(mask_height, mask_width, 4, device=device, dtype=torch.uint8)
    
    # Process using smart batching
    total_patches = mask_height * mask_width
    print(f"Total patches to process: {total_patches}")
    print(f"Mask dimensions: {mask_width}x{mask_height}")
    
    batch_num = 0
    processed_patches = 0
    ow_count = 0
    tw_count = 0
    update_interval = max(1, total_patches // 100)  # Update ~100 times total
    
    with torch.no_grad():
        for batch, positions in smart_batched_sliding_window(img_tensor, window_size, device, net):
            batch_num += 1
            processed_patches += len(batch)
            
            # Only print progress every N batches or when significant progress is made
            if batch_num == 1 or processed_patches % update_interval < len(batch) or processed_patches == total_patches:
                percentage = (processed_patches / total_patches) * 100
                print(f"Progress: {processed_patches}/{total_patches} ({percentage:.1f}%) - Batch {batch_num}")
            
            outputs = net(batch)
            predictions = torch.argmax(outputs, dim=1)
            
            # Count predictions for debugging - only show on significant updates
            batch_ow = torch.sum(predictions == 0).item()
            batch_tw = torch.sum(predictions == 1).item()
            ow_count += batch_ow
            tw_count += batch_tw
            
            if batch_num == 1 or processed_patches % update_interval < len(batch) or processed_patches == total_patches:
                print(f"  Batch predictions - OW: {batch_ow}, TW: {batch_tw}")
                print(f"  Cumulative - OW: {ow_count}, TW: {tw_count}")
            
            # Update mask on GPU - set one pixel per window at the corresponding position in the mask
            for i, (y, x) in enumerate(positions):
                pred = predictions[i].item()
                
                # The y,x coordinates are already within the mask's dimensions
                if pred == 0:  # OW
                    # Set blue channel and alpha
                    mask_tensor[y, x, 2] = 255  # Blue
                    mask_tensor[y, x, 3] = 255  # Alpha
                else:  # TW
                    # Set red channel and alpha
                    mask_tensor[y, x, 0] = 255  # Red
                    mask_tensor[y, x, 3] = 255  # Alpha
    
    print(f"Total predictions - OW: {ow_count}, TW: {tw_count}")

    # Move mask to CPU and save
    mask = mask_tensor.cpu().numpy()
    mask_img = Image.fromarray(mask, "RGBA")
    # Generate output filename
    base_name = os.path.splitext(args.jpeg_file)[0]
    output_filename = f"{base_name}_mask.png"
    mask_img.save(output_filename)
    print(f"Mask saved to: {output_filename}")


if __name__ == "__main__":
    main()
