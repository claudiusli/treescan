from torch.utils.data import IterableDataset, DataLoader
import random
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim


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
                label = "ow"
            else:
                file_path = self.tw_file
                label = "tw"

            # Open the image
            with Image.open(file_path) as img:
                width, height = img.size

                # Ensure the window is valid
                if width < self.window or height < self.window:
                    raise ValueError(
                        f"Image dimensions {width}x{height} are smaller than window size {self.window}"
                    )

                # Generate random coordinates
                x = random.randint(0, width - self.window)
                y = random.randint(0, height - self.window)

                # Extract the patch
                patch = img.crop((x, y, x + self.window, y + self.window))

                # Convert to tensor or appropriate format
                # For now, we'll yield the PIL Image and label, but you might want to convert to tensor
                yield patch, label


def pil_collate_fn(batch):
    """
    Custom collate function to handle PIL Images in batches.
    Converts PIL Images to tensors and stacks them.
    """
    images, labels = zip(*batch)

    # Convert PIL Images to tensors
    image_tensors = []
    for img in images:
        # Convert PIL Image to tensor
        if isinstance(img, Image.Image):
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Convert to tensor
            img_tensor = torch.tensor(list(img.getdata())).view(
                img.size[1], img.size[0], 3
            )
            img_tensor = img_tensor.permute(2, 0, 1)  # Change to (C, H, W)
            image_tensors.append(img_tensor)
        else:
            # If it's already a tensor, just use it
            image_tensors.append(img)

    # Stack images and labels
    images_stacked = torch.stack(image_tensors)
    labels_stacked = list(labels)

    return images_stacked, labels_stacked



# Create train and test datasets
def stream(ow_file, tw_file, window, train_size, test_size):
    # Validate files exist
    if not os.path.exists(ow_file):
        raise FileNotFoundError(f"OW file not found: {ow_file}")
    if not os.path.exists(tw_file):
        raise FileNotFoundError(f"TW file not found: {tw_file}")

    # Create datasets
    train_ds = StreamDS(ow_file, tw_file, window, train_size)
    test_ds = StreamDS(ow_file, tw_file, window, test_size)

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


# Example usage
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
        required=True,
        help="Number of samples in the training dataset",
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

    # Parse arguments
    args = parser.parse_args()

    # Create datasets using provided arguments
    train, test = stream(
        args.ow_file, args.tw_file, args.window, args.train_size, args.test_size
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train, batch_size=args.batch_size, num_workers=0, collate_fn=pil_collate_fn
    )
    test_loader = DataLoader(
        test, batch_size=args.batch_size, num_workers=0, collate_fn=pil_collate_fn
    )

    # Define the network
    class SimpleCNN(nn.Module):
        def __init__(self, window_size):
            super(SimpleCNN, self).__init__()
            self.window_size = window_size
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            # Calculate the size after convolutions and pooling
            # Each pooling layer reduces by half, so two poolings: window_size // 4
            self.fc1 = nn.Linear(64 * (window_size // 4) * (window_size // 4), 64)
            self.fc2 = nn.Linear(64, 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * (self.window_size // 4) * (self.window_size // 4))
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the network, loss function, and optimizer
    net = SimpleCNN(args.window)
    net.to(device)  # Move network to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # Move inputs and labels to GPU
            inputs = inputs.float().to(device)
            
            # Convert labels to tensor indices and move to GPU
            label_indices = [0 if label == 'ow' else 1 for label in labels]
            label_tensor = torch.tensor(label_indices, device=device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    
    # Testing loop
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # Move images to GPU
            images = images.float().to(device)
            
            # Convert labels to tensor indices and move to GPU
            label_indices = [0 if label == 'ow' else 1 for label in labels]
            label_tensor = torch.tensor(label_indices, device=device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
