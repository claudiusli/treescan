import torch
from PIL import Image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def load_model_window_and_network(model_path, device):
    """Load the model file and extract the window size and network"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the saved data
    save_data = torch.load(model_path, map_location=device)
    
    if 'window_size' not in save_data:
        raise ValueError("Model file does not contain window_size information")
    
    window_size = save_data['window_size']
    
    # Define the network architecture (must match classifier.py)
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
    
    # Initialize the network
    net = StainDiscriminator(window_size)
    net.load_state_dict(save_data['model_state_dict'])
    net.to(device)
    net.eval()  # Set to evaluation mode
    
    return window_size, net

def load_image_to_tensor(image_path, device):
    """Load the image and convert to tensor on the specified device, matching classifier.py"""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Convert to tensor and move to device, matching classifier.py
        tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()
        return tensor.to(device)

class PatchViewer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def show_patch(self, image_tensor, x, y, window_size, net, device):
        """Extract and display the window x window patch at (x, y) in a matplotlib window, and predict OW/TW"""
        # Get image dimensions from the tensor (C, H, W)
        _, height, width = image_tensor.shape
        
        # Extract patch directly from the tensor, matching classifier.py
        # The tensor is on the specified device
        patch_tensor = image_tensor[:, y:y + window_size, x:x + window_size]
        
        # Add batch dimension for the network
        patch_batch = patch_tensor.unsqueeze(0)
        
        # Get prediction from the network
        with torch.no_grad():
            outputs = net(patch_batch)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get the predicted class (0: OW, 1: TW)
            _, predicted = torch.max(outputs, 1)
            
            # Convert to percentages
            ow_prob = probabilities[0][0].item() * 100
            tw_prob = probabilities[0][1].item() * 100
            label = 'OW' if predicted.item() == 0 else 'TW'
        
        # Convert the patch tensor to numpy for display
        # The tensor is on CPU or GPU, so move to CPU and convert
        patch_np = patch_tensor.cpu().permute(1, 2, 0).numpy()
        # Convert to uint8 for display
        patch_np = np.clip(patch_np, 0, 255).astype(np.uint8)
        
        # Close previous figure if it exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create a new figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.ax.imshow(patch_np)
        self.ax.set_title(f'Patch at ({x}, {y})\nPrediction: {label} (OW: {ow_prob:.2f}%, TW: {tw_prob:.2f}%)')
        self.ax.axis('off')
        
        # Use plt.show(block=False) to not block the main thread
        plt.show(block=False)
        # Add a small pause to ensure the window is drawn
        plt.pause(0.1)
        
        return width, height

def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze.py <image_path> <model_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load window size and network from model
    try:
        window_size, net = load_model_window_and_network(model_path, device)
        print(f"Window size: {window_size}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load image as tensor on the device
    try:
        image_tensor = load_image_to_tensor(image_path, device)
        print(f"Image tensor shape: {image_tensor.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Get image dimensions from the tensor (C, H, W)
    _, height, width = image_tensor.shape
    
    # Check if window size is valid
    if window_size > width or window_size > height:
        print(f"Window size {window_size} is larger than image dimensions {width}x{height}")
        sys.exit(1)
    
    # Initialize the patch viewer
    viewer = PatchViewer()
    
    # Start at (0, 0)
    x, y = 0, 0
    
    while True:
        print(f"Current position: ({x}, {y})")
        # Show the current patch and get prediction
        width, height = viewer.show_patch(image_tensor, x, y, window_size, net, device)
        
        # Get user input
        try:
            user_input = input("Enter number of pixels to move right (negative for left), or 'q' to quit: ")
            if user_input.lower() == 'q':
                # Close the matplotlib window when quitting
                if viewer.fig is not None:
                    plt.close(viewer.fig)
                break
            
            move = int(user_input)
            
            # Calculate new x position
            new_x = x + move
            new_y = y
            
            # Handle wrapping within the current row
            if new_x < 0:
                # Move to previous row
                new_y -= window_size
                new_x = width - window_size + new_x  # new_x is negative, so this subtracts
                # Check if we've gone above the image
                if new_y < 0:
                    print("Cannot move beyond the top-left corner of the image")
                    continue
            elif new_x > width - window_size:
                # Move to next row
                new_y += window_size
                new_x = new_x - (width - window_size) - 1
                # Check if we've gone below the image
                if new_y > height - window_size:
                    print("Cannot move beyond the bottom-right corner of the image")
                    continue
            else:
                # The new x is within bounds, y remains the same
                pass
            
            # Check if the new position is valid
            if (0 <= new_x <= width - window_size and 
                0 <= new_y <= height - window_size):
                x, y = new_x, new_y
            else:
                print("Invalid move: would go beyond image boundaries")
                continue
                
        except ValueError:
            print("Please enter a valid integer or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            # Close the matplotlib window when quitting
            if viewer.fig is not None:
                plt.close(viewer.fig)
            break

if __name__ == "__main__":
    main()
