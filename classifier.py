from torch.utils.data import IterableDataset, DataLoader
import random
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class StreamDS(IterableDataset):
    def __init__(self, ow_file, tw_file, window, size, device):
        self.ow_file = ow_file
        self.tw_file = tw_file
        self.window = window
        self.size = size
        self.device = device
        # Pre-load the images to GPU memory
        self.ow_img = self._load_image_to_gpu(ow_file, device)
        self.tw_img = self._load_image_to_gpu(tw_file, device)

    def _load_image_to_gpu(self, path, device):
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Convert to tensor and move to GPU
            tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()
            return tensor.to(device)

    def __iter__(self):
        for _ in range(self.size):
            # Randomly choose which image tensor to use
            if random.choice([True, False]):
                img_tensor = self.ow_img
                label = "ow"
            else:
                img_tensor = self.tw_img
                label = "tw"

            # Get random coordinates
            _, height, width = img_tensor.shape
            if width < self.window or height < self.window:
                raise ValueError(
                    f"Image dimensions {width}x{height} are smaller than window size {self.window}"
                )
            x = random.randint(0, width - self.window)
            y = random.randint(0, height - self.window)

            # Extract patch directly from GPU tensor
            patch = img_tensor[:, y : y + self.window, x : x + self.window]

            yield patch, label


def gpu_collate_fn(batch, device):
    patches, labels = zip(*batch)
    # Stack patches (already tensors on GPU)
    patches_stacked = torch.stack(patches)
    # Convert labels to tensor
    label_indices = [0 if label == "ow" else 1 for label in labels]
    labels_stacked = torch.tensor(label_indices, device=device)
    return patches_stacked, labels_stacked


# Create train and test datasets
def stream(ow_file, tw_file, window, train_size, test_size, device):
    # Validate files exist
    if not os.path.exists(ow_file):
        raise FileNotFoundError(f"OW file not found: {ow_file}")
    if not os.path.exists(tw_file):
        raise FileNotFoundError(f"TW file not found: {tw_file}")

    # Create datasets with device
    train_ds = StreamDS(ow_file, tw_file, window, train_size, device)
    test_ds = StreamDS(ow_file, tw_file, window, test_size, device)

    return train_ds, test_ds


def examine(loader):
    print("=== Examining DataLoader ===")

    # Track some metadata
    total_batches = 0
    total_samples = 0
    label_counts = {"ow": 0, "tw": 0}
    image_shapes = set()

    # Examine a few batches to gather information
    for i, batch in enumerate(loader):
        if i >= 5:  # Only look at first 5 batches to avoid consuming too much
            break

        total_batches += 1
        # Unpack the batch
        samples, labels = batch
        total_samples += len(samples)

        # Count labels
        for label in labels:
            if label == "ow":
                label_counts["ow"] += 1
            elif label == "tw":
                label_counts["tw"] += 1

        # Record image shapes (now tensors)
        for sample in samples:
            if hasattr(sample, "shape"):
                image_shapes.add(tuple(sample.shape))

    # Print the gathered metadata
    print(f"Number of batches examined: {total_batches}")
    print(f"Total samples examined: {total_samples}")
    print(f"Label distribution: {label_counts}")
    print(f"Image shapes: {image_shapes}")

    # Calculate percentages
    if total_samples > 0:
        ow_percent = (label_counts["ow"] / total_samples) * 100
        tw_percent = (label_counts["tw"] / total_samples) * 100
        print(f"Label percentages: OW: {ow_percent:.2f}%, TW: {tw_percent:.2f}%")

    print("Note: Only examined first 5 batches to avoid consuming the entire iterator")


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create train and test datasets for image classification"
    )
    parser.add_argument(
        "--ow_file", type=str, required=True, help="Path to the OW image file"
    )
    parser.add_argument(
        "--tw_file", type=str, required=True, help="Path to the TW image file"
    )
    parser.add_argument(
        "--window", type=int, required=True, help="Window size for cropping patches"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=0,
        help="Number of samples in the training dataset (default: 0)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=True,
        help="Number of samples in the test dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for DataLoader (default: 64)",
    )
    parser.add_argument(
        "--out_model",
        type=str,
        default="model.weights",
        help="Path to save the trained model (default: model.weights)",
    )
    parser.add_argument(
        "--in_model",
        type=str,
        default="",
        help="Path to load pre-trained model weights (optional)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets using provided arguments
    train, test = stream(
        args.ow_file, args.tw_file, args.window, args.train_size, args.test_size, device
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=lambda batch: gpu_collate_fn(batch, device),
    )
    test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=lambda batch: gpu_collate_fn(batch, device),
    )

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define a network optimized for color-based stain discrimination
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

    # Initialize the network, loss function, and optimizer
    net = StainDiscriminator(args.window)
    net.to(device)  # Move network to GPU
    
    # Load pre-trained weights if specified
    if args.in_model:
        if not os.path.exists(args.in_model):
            raise FileNotFoundError(f"Model file not found: {args.in_model}")
        
        # Load the saved data
        save_data = torch.load(args.in_model, map_location=device)
        
        # Check if window size matches
        if 'window_size' in save_data:
            if save_data['window_size'] != args.window:
                print(f"Error: Window size mismatch! Model was trained with window size {save_data['window_size']}, "
                      f"but current window size is {args.window}")
                exit(1)
        else:
            print("Warning: No window size information found in the model file. Proceeding anyway.")
        
        # Load the model state dict
        net.load_state_dict(save_data['model_state_dict'])
        print(f"Loaded pre-trained weights from '{args.in_model}'")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Only train if train_size > 0
    if args.train_size > 0:
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Data is already on the GPU from the collate function
                inputs, label_tensor = data

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, label_tensor)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(
                        f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.8f}"
                    )
                    running_loss = 0.0

        print("Finished Training")
    else:
        print("Skipping training (train_size is 0)")

    # Save the model weights along with the window size
    if args.train_size > 0:
        save_data = {
            'model_state_dict': net.state_dict(),
            'window_size': args.window
        }
        torch.save(save_data, args.out_model)
        print(f"Model weights and window size saved to '{args.out_model}'")
    else:
        print("Skipping model saving (train_size is 0)")

    # Testing loop
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, label_tensor = data
            # Data is already on the GPU from the collate function

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.8f}%")
