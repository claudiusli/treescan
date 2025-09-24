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

def main():
    parser = argparse.ArgumentParser(description='Create a mask from a JPEG image using a trained classifier')
    parser.add_argument('jpeg_file', type=str, help='Path to the JPEG image file')
    parser.add_argument('model_file', type=str, help='Path to the trained model weights file')
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
    window_size = save_data['window_size']
    print(f"Window size: {window_size}")
    
    # Initialize the network
    net = StainDiscriminator(window_size)
    net.load_state_dict(save_data['model_state_dict'])
    net.to(device)
    net.eval()  # Set to evaluation mode
    
    # Load and process the JPEG image
    with Image.open(args.jpeg_file) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        height, width, _ = img_array.shape
    
    # Create an output image for the mask (RGBA to support transparency)
    # We'll make OW regions blue and TW regions red
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Process each window
    with torch.no_grad():
        for y in range(0, height - window_size + 1):
            for x in range(0, width - window_size + 1):
                # Extract the patch
                patch = img_array[y:y+window_size, x:x+window_size, :]
                # Convert to tensor and preprocess
                patch_tensor = torch.tensor(patch).permute(2, 0, 1).float().unsqueeze(0)
                patch_tensor = patch_tensor.to(device)
                
                # Classify
                outputs = net(patch_tensor)
                _, predicted = torch.max(outputs, 1)
                classification = predicted.item()
                
                # Color the region in the mask
                # OW (class 0) -> Blue, TW (class 1) -> Red
                if classification == 0:  # OW
                    # Set blue channel and alpha
                    mask[y:y+window_size, x:x+window_size, 2] = 255  # Blue
                    mask[y:y+window_size, x:x+window_size, 3] = 128  # Alpha (semi-transparent)
                else:  # TW
                    # Set red channel and alpha
                    mask[y:y+window_size, x:x+window_size, 0] = 255  # Red
                    mask[y:y+window_size, x:x+window_size, 3] = 128  # Alpha (semi-transparent)
    
    # Save the mask
    mask_img = Image.fromarray(mask, 'RGBA')
    # Generate output filename
    base_name = os.path.splitext(args.jpeg_file)[0]
    output_filename = f"{base_name}_mask.png"
    mask_img.save(output_filename)
    print(f"Mask saved to: {output_filename}")

if __name__ == "__main__":
    main()
