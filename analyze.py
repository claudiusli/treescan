import torch
from PIL import Image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def load_model_window(model_path):
    """Load the model file and extract the window size"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the saved data
    save_data = torch.load(model_path, map_location='cpu')
    
    if 'window_size' not in save_data:
        raise ValueError("Model file does not contain window_size information")
    
    return save_data['window_size']

def load_image(image_path):
    """Load the image and return as a PIL Image"""
    return Image.open(image_path)

class PatchViewer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def show_patch(self, image, x, y, window_size):
        """Extract and display the window x window patch at (x, y) in a matplotlib window"""
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # Extract the patch
        patch = image.crop((x, y, x + window_size, y + window_size))
        
        # Convert to numpy array for matplotlib
        patch_array = np.array(patch)
        
        # Close previous figure if it exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create a new figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.ax.imshow(patch_array)
        self.ax.set_title(f'Patch at ({x}, {y})')
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
    
    # Load window size from model
    try:
        window_size = load_model_window(model_path)
        print(f"Window size: {window_size}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load image
    try:
        image = load_image(image_path)
        print(f"Image size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Get image dimensions
    width, height = image.size
    
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
        # Show the current patch
        width, height = viewer.show_patch(image, x, y, window_size)
        
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
