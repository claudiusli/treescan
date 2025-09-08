from torch.utils.data import IterableDataset, DataLoader
import random
from PIL import Image
import os


class StreamDS(IterableDataset):
    def __init__(self, ow_file, tw_file, window, size):
        self.ow_file = ow_file
        self.tw_file = tw_file
        self.window = window
        self.size = size
        
    def __iter__(self):
        for _ in range(self.size):
            # Randomly choose which file to use
            if random.choice([True, False]):
                file_path = self.ow_file
                label = 'ow'
            else:
                file_path = self.tw_file
                label = 'tw'
            
            # Open the image
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Ensure the window is valid
                if width < self.window or height < self.window:
                    raise ValueError(f"Image dimensions {width}x{height} are smaller than window size {self.window}")
                
                # Generate random coordinates
                x = random.randint(0, width - self.window)
                y = random.randint(0, height - self.window)
                
                # Extract the patch
                patch = img.crop((x, y, x + self.window, y + self.window))
                
                # Convert to tensor or appropriate format
                # For now, we'll yield the PIL Image and label, but you might want to convert to tensor
                yield patch, label


# Create train and test datasets
def main(ow_file, tw_file, window, train_size, test_size):
    # Validate files exist
    if not os.path.exists(ow_file):
        raise FileNotFoundError(f"OW file not found: {ow_file}")
    if not os.path.exists(tw_file):
        raise FileNotFoundError(f"TW file not found: {tw_file}")
    
    # Create datasets
    train_ds = StreamDS(ow_file, tw_file, window, train_size)
    test_ds = StreamDS(ow_file, tw_file, window, test_size)
    
    return train_ds, test_ds


# Example usage
if __name__ == "__main__":
    # Parameters
    ow_file = "ow.png"
    tw_file = "tw.png"
    window = 64
    train_size = 1000
    test_size = 200
    
    train, test = main(ow_file, tw_file, window, train_size, test_size)
    
    # Create data loaders
    train_loader = DataLoader(train, batch_size=64, num_workers=0)
    test_loader = DataLoader(test, batch_size=64, num_workers=0)
