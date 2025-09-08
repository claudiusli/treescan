from torch.utils.data import IterableDataset, DataLoader
import random
from PIL import Image
import os
import torch


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # examine(train_loader)
    # examine(test_loader)
