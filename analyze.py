import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import sys
import os

# Define the same network architecture as in classifier.py
class StainDiscriminator(nn.Module):
    def __init__(self, window_size):
        super(StainDiscriminator, self).__init__()
        self.color_attention = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )
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
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, x):
        color_weights = self.color_attention(x)
        x = x * color_weights
        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        return self.classifier(flattened)

def calculate_max_batch_size(window_size, device, net):
    # Estimate the maximum batch size that fits in GPU memory
    # Start with a conservative batch size and increase until we hit memory limits
    torch.cuda.empty_cache()
    batch_size = 1
    while True:
        try:
            # Create a dummy batch to test memory usage
            dummy_input = torch.randn(batch_size, 3, window_size, window_size).to(device)
            # Forward pass
            net(dummy_input)
            # If successful, try next batch size
            batch_size *= 2
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                # Use half the last successful batch size to be safe
                batch_size = max(batch_size // 4, 1)
                break
    return batch_size

def smart_batched_sliding_window(tensor, window_size, device, net):
    _, height, width = tensor.shape
    # Calculate maximum batch size
    max_batch_size = calculate_max_batch_size(window_size, device, net)
    
    # Generate all window coordinates
    x_coords = []
    y_coords = []
    x = 0
    y = 0
    while True:
        if x + window_size > width:
            x = 0
            y += 1
            if y + window_size > height:
                break
            continue
        x_coords.append(x)
        y_coords.append(y)
        x += 1
        if x + window_size > width:
            x = 0
            y += 1
            if y + window_size > height:
                break
    
    # Process in batches
    net.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(x_coords), max_batch_size):
            # Get current batch coordinates
            batch_x = x_coords[i:i + max_batch_size]
            batch_y = y_coords[i:i + max_batch_size]
            
            # Prepare batch
            batch_tensors = []
            for x, y in zip(batch_x, batch_y):
                patch = tensor[:, y:y + window_size, x:x + window_size]
                batch_tensors.append(patch)
            
            # Stack into a batch
            batch = torch.stack(batch_tensors)
            
            # Forward pass
            outputs = net(batch)
            _, predicted = torch.max(outputs, 1)
            
            # Convert to labels
            for pred in predicted:
                label = 'TW' if pred.item() == 1 else 'OW'
                predictions.append(label)
            
            # Print progress
            print(f'Processed {min(i + max_batch_size, len(x_coords))}/{len(x_coords)} windows')
    
    return predictions, x_coords, y_coords

def analyze_image(image_path: str, model_path: str) -> None:
    # Check if files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    save_data = torch.load(model_path, map_location=device)
    window_size = save_data['window_size']
    print(f"Window size: {window_size}")
    
    # Initialize network
    net = StainDiscriminator(window_size)
    net.to(device)
    net.load_state_dict(save_data['model_state_dict'])
    net.eval()
    
    # Load and prepare image
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Convert to tensor and move to GPU
        img_array = np.array(img)
        tensor = torch.tensor(img_array).permute(2, 0, 1).float().to(device)
    
    # Process using sliding window
    predictions, x_coords, y_coords = smart_batched_sliding_window(tensor, window_size, device, net)
    
    # Output results
    for x, y, pred in zip(x_coords, y_coords, predictions):
        print(f"{x},{y}: {pred}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze.py <image_path> <model_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    analyze_image(image_path, model_path)

if __name__ == "__main__":
    main()
