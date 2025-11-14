import json
import sys
from pathlib import Path
import random
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from .image_operations import load_ppm_image, parse_json_params


def prepare_training_data(samples_dir):
    """Load and prepare training data from samples directory"""
    if not samples_dir.exists():
        raise ValueError(f"Samples directory does not exist: {samples_dir}")
    
    sample_files = [f for f in samples_dir.glob('*.*') if not f.name.endswith('.model')]
    
    if len(sample_files) == 0:
        raise ValueError(f"No sample files found in {samples_dir}")
    
    # Group files by color
    color_files = {}
    for file_path in sample_files:
        color = file_path.suffix[1:]
        if color not in color_files:
            color_files[color] = []
        color_files[color].append(file_path)
    
    if len(color_files) != 2:
        raise ValueError(f"Expected exactly 2 colors, found {len(color_files)}: {list(color_files.keys())}")
    
    return color_files, sorted(color_files.keys())


def select_random_sample_file(color_files, color):
    """Select a random file for the given color"""
    return random.choice(color_files[color])

def extract_random_window(img, window_size):
    """Extract a random window from image"""
    max_x = img.width - window_size
    max_y = img.height - window_size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return img.crop((x, y, x + window_size, y + window_size))

def create_batch_tensors(images, labels, device):
    """Convert lists to tensors on device"""
    if images:
        return (torch.stack(images).to(device), 
                torch.tensor(labels, dtype=torch.long).to(device))
    return None, None

def create_training_batch(color_files, colors, window_size, device):
    """Create a single training batch"""
    batch_images = []
    batch_labels = []
    
    for color_idx, color in enumerate(colors):
        file_path = select_random_sample_file(color_files, color)
        img = load_ppm_image(file_path, require_ppm=True)
        
        if img.width < window_size or img.height < window_size:
            continue
            
        window = extract_random_window(img, window_size)
        tensor = transforms.ToTensor()(window)
        batch_images.append(tensor)
        batch_labels.append(color_idx)
    
    return create_batch_tensors(batch_images, batch_labels, device)

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
        
        # Enhanced feature extraction matching classifier.py
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),  # Added BatchNorm for stability
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Explicit stride
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Explicit stride
            nn.Conv2d(64, 128, 3, padding=1),  # Added third conv layer
            nn.BatchNorm2d(128),  # Added BatchNorm
            nn.ReLU(),
        )
        
        # Global average pooling for spatial invariance
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(128, 32), 
            nn.ReLU(), 
            nn.Dropout(0.3),  # Added dropout for regularization
            nn.Linear(32, 2)   # Binary classification
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

def create_model(window_size, device):
    """Create a new StainDiscriminator model"""
    model = StainDiscriminator(window_size).to(device)
    return model

def load_model(model_path, device):
    """Load an existing model from file"""
    if not Path(model_path).exists():
        raise ValueError(f"Model file does not exist: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    window_size = checkpoint['window_size']
    colors = checkpoint['colors']
    metadata = checkpoint.get('metadata', {'training_history': []})
    
    model = create_model(window_size, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, window_size, colors, metadata

def save_model(model, model_path, window_size, colors, metadata):
    """Save model to file with metadata"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'window_size': window_size,
        'colors': colors,
        'metadata': metadata
    }, model_path)

def get_device():
    """Get the appropriate device (CUDA or CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def setup_training_params(params):
    """Extract and validate training parameters"""
    samples_dir = Path(params['samples'])
    window_size = params['window']
    train_count = params['traincount']
    return samples_dir, window_size, train_count

def setup_model(window_size, device, model_path=None):
    """Setup model - create new or load existing"""
    if model_path and Path(model_path).exists():
        print(f"Loading existing model from {model_path}")
        model, loaded_window_size, loaded_colors, metadata = load_model(model_path, device)
        
        if loaded_window_size != window_size:
            raise ValueError(f"Model window size ({loaded_window_size}) doesn't match requested size ({window_size})")
        return model, loaded_colors, metadata
    else:
        print(f"Creating new model with window size {window_size}")
        model = create_model(window_size, device)
        metadata = {'training_history': []}
        return model, None, metadata

def setup_training(model, learning_rate=0.001):
    """Setup optimizer and loss function matching classifier.py"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

def train_single_batch(model, batch_images, batch_labels, criterion, optimizer):
    """Train on a single batch"""
    optimizer.zero_grad()
    outputs = model(batch_images)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
    return loss

def should_print_progress(iteration, start_time, last_print_time, print_interval=1.0):
    """Determine if we should print progress"""
    current_time = time.time()
    return current_time - last_print_time >= print_interval

def run_training_loop(model, color_files, colors, window_size, device, criterion, optimizer, train_count):
    """Run the main training loop"""
    model.train()
    print(f"Starting training for {train_count} iterations...")
    
    start_time = time.time()
    last_print_time = start_time
    
    for iteration in range(train_count):
        batch_images, batch_labels = create_training_batch(color_files, colors, window_size, device)
        
        if batch_images is None:
            continue
        
        loss = train_single_batch(model, batch_images, batch_labels, criterion, optimizer)
        
        # Print progress
        if should_print_progress(iteration, start_time, last_print_time):
            print(f"Iteration {iteration + 1}/{train_count}, Loss: {loss.item():.4f}")
            last_print_time = time.time()
    
    return loss

def create_training_metadata(train_count, window_size, colors, final_loss):
    """Create training metadata entry"""
    return {
        'timestamp': datetime.now().isoformat(),
        'traincount': train_count,
        'window_size': window_size,
        'colors': colors,
        'final_loss': final_loss
    }

def train_model(json_str, model_path=None):
    """Train a neural network on samples in the specified directory"""
    params = parse_json_params(json_str, ['samples', 'window', 'traincount'], 'train')
    samples_dir, window_size, train_count = setup_training_params(params)
    
    # Prepare training data
    color_files, colors = prepare_training_data(samples_dir)
    print(f"Training on colors: {colors}")
    
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create or load model
    model, loaded_colors, metadata = setup_model(window_size, device, model_path)
    
    # If we loaded a model, use its colors, otherwise use the ones from the samples
    if loaded_colors is not None:
        colors = loaded_colors
    
    # Training setup
    criterion, optimizer = setup_training(model)
    
    # Run training
    final_loss = run_training_loop(model, color_files, colors, window_size, device, 
                                 criterion, optimizer, train_count)
    
    # Save model
    model_filename = f"{window_size}.model"
    model_save_path = samples_dir / model_filename
    
    training_entry = create_training_metadata(train_count, window_size, colors, final_loss.item())
    metadata['training_history'].append(training_entry)
    
    save_model(model, model_save_path, window_size, colors, metadata)
    print(f"Training completed. Model saved to: {model_save_path}")
    print(f"Final loss: {final_loss.item():.4f}")

def test_model(json_str, model_path):
    """Test a neural network model on samples"""
    if not model_path:
        print("Error: --model parameter is required for --test")
        print("Usage: --test '{\"samples\":\"<string>\",\"window\":<integer>,\"testcount\":<integer>}' --model <model_file>")
        sys.exit(1)
    
    params = parse_json_params(json_str, ['samples', 'window', 'testcount'], 'test')
    print("Test functionality not yet implemented")
    # TODO: Implement test functionality

def makemask(json_str, model_path):
    """Create a mask bitmap using the neural network model"""
    if not model_path:
        print("Error: --model parameter is required for --makemask")
        print("Usage: --makemask '{\"image\":\"<string>\"}' --model <model_file>")
        sys.exit(1)
    
    params = parse_json_params(json_str, ['image'], 'makemask')
    print("Makemask functionality not yet implemented")
    # TODO: Implement makemask functionality
