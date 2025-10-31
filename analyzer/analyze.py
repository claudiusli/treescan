#!/usr/bin/env python3

import argparse
import json
import sys
import os
from pathlib import Path
from PIL import Image
import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import random
import time
import yaml
from datetime import datetime

def normalize_image(image_path):
    """Convert an image to PPM P6 format maintaining original color depth"""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None
        
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if not already (PPM P6 requires RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create output path with .ppm extension
            input_path = Path(image_path)
            output_path = input_path.parent / (input_path.stem + '.ppm')
            
            # Save as PPM P6 format
            img.save(output_path, format='PPM')
            
            print(f"Normalized image saved to: {output_path}")
            
    except Exception as e:
        print(f"Error normalizing image: {e}")
        sys.exit(1)

def makesample(json_str):
    """Create a sample subimage from the specified coordinates"""
    try:
        # Parse JSON input
        params = json.loads(json_str)
        required_keys = ['image', 'x', 'y', 'w', 'h', 'color']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
        
        image_path = params['image']
        x, y, w, h = params['x'], params['y'], params['w'], params['h']
        color = params['color']
        
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None
        
        # Open the image - must be PPM or throw error
        try:
            with Image.open(image_path) as img:
                # Check if it's a PPM file
                if img.format != 'PPM':
                    raise ValueError(f"Image must be in PPM format, got {img.format}")
                
                # Validate coordinates
                if x < 0 or y < 0 or x + w > img.width or y + h > img.height:
                    raise ValueError(f"Sample coordinates ({x}, {y}, {w}, {h}) exceed image bounds ({img.width}, {img.height})")
                
                # Create samples directory
                input_path = Path(image_path)
                samples_dir = input_path.parent / (input_path.stem + '.samples')
                samples_dir.mkdir(exist_ok=True)
                
                # Extract subimage
                subimage = img.crop((x, y, x + w, y + h))
                
                # Create output filename: <x>_<y>_<w>_<h>.<color>
                sample_filename = f"{x}_{y}_{w}_{h}.{color}"
                sample_path = samples_dir / sample_filename
                
                # Save as PPM maintaining same color depth
                subimage.save(sample_path, format='PPM')
                
                print(f"Sample saved to: {sample_path}")
                
        except Exception as e:
            if "cannot identify image file" in str(e).lower():
                raise ValueError(f"Cannot read image file: {image_path}")
            raise
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating sample: {e}")
        sys.exit(1)

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
        
        # Spatial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate the size after convolutions
        feature_size = 64 * (window_size // 4) * (window_size // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification
        )
        
    def forward(self, x):
        # Apply color attention
        attention = self.color_attention(x)
        x = x * attention
        
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        return x

def train_model(json_str, model_path=None):
    """Train a neural network on samples in the specified directory"""
    try:
        # Parse JSON input
        params = json.loads(json_str)
        required_keys = ['samples', 'window', 'traincount']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
        
        samples_dir = Path(params['samples'])
        window_size = params['window']
        train_count = params['traincount']
        
        if not samples_dir.exists():
            raise ValueError(f"Samples directory does not exist: {samples_dir}")
        
        # Get all sample files (ignore .model files)
        sample_files = [f for f in samples_dir.glob('*.*') if not f.name.endswith('.model')]
        
        if len(sample_files) == 0:
            raise ValueError(f"No sample files found in {samples_dir}")
        
        # Extract colors from filenames and group files by color
        color_files = {}
        for file_path in sample_files:
            color = file_path.suffix[1:]  # Remove the dot
            if color not in color_files:
                color_files[color] = []
            color_files[color].append(file_path)
        
        if len(color_files) != 2:
            raise ValueError(f"Expected exactly 2 colors, found {len(color_files)}: {list(color_files.keys())}")
        
        colors = sorted(color_files.keys())  # Sort for consistency
        print(f"Training on colors: {colors}")
        
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create or load model
        if model_path and Path(model_path).exists():
            print(f"Loading existing model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model = StainDiscriminator(window_size).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            metadata = checkpoint.get('metadata', {'training_history': []})
        else:
            print(f"Creating new model with window size {window_size}")
            model = StainDiscriminator(window_size).to(device)
            metadata = {'training_history': []}
        
        # Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        # Training loop
        print(f"Starting training for {train_count} iterations...")
        start_time = time.time()
        last_print_time = start_time
        
        for iteration in range(train_count):
            # Randomly select one file from each color
            batch_images = []
            batch_labels = []
            
            for color_idx, color in enumerate(colors):
                # Randomly select a file of this color
                file_path = random.choice(color_files[color])
                
                # Load image
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Get image dimensions
                    img_width, img_height = img.size
                    
                    # Ensure we can extract a full window
                    if img_width < window_size or img_height < window_size:
                        continue
                    
                    # Randomly select a window position
                    max_x = img_width - window_size
                    max_y = img_height - window_size
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    
                    # Extract window
                    window = img.crop((x, y, x + window_size, y + window_size))
                    
                    # Convert to tensor
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    
                    tensor = transform(window)
                    batch_images.append(tensor)
                    batch_labels.append(color_idx)
            
            if len(batch_images) == 0:
                continue
            
            # Create batch
            batch_images = torch.stack(batch_images).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Print loss approximately once per second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                print(f"Iteration {iteration + 1}/{train_count}, Loss: {loss.item():.4f}")
                last_print_time = current_time
        
        # Save model
        model_filename = f"{window_size}.model"
        model_path = samples_dir / model_filename
        
        # Update metadata
        training_entry = {
            'timestamp': datetime.now().isoformat(),
            'traincount': train_count,
            'window_size': window_size,
            'colors': colors,
            'sample_files': [str(f) for f in sample_files],
            'final_loss': loss.item()
        }
        metadata['training_history'].append(training_entry)
        
        # Save model with metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'window_size': window_size,
            'colors': colors,
            'metadata': metadata
        }, model_path)
        
        print(f"Training completed. Model saved to: {model_path}")
        print(f"Final loss: {loss.item():.4f}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def run_unit_tests():
    """Run unit tests to verify all functionality works"""
    
    class TestAnalyzer(unittest.TestCase):
        
        def setUp(self):
            """Set up test fixtures"""
            self.test_dir = Path(tempfile.mkdtemp())
            
            # Create a small test image (10x10 RGB)
            self.test_image = Image.new('RGB', (10, 10), color='red')
            self.test_image_path = self.test_dir / 'test_image.jpg'
            self.test_image.save(self.test_image_path)
            
        def tearDown(self):
            """Clean up test fixtures"""
            shutil.rmtree(self.test_dir)
            
        def test_normalize_image(self):
            """Test that normalize_image creates proper PPM file"""
            # Test the normalize function
            normalize_image(str(self.test_image_path))
            
            # Check that PPM file was created
            expected_ppm = self.test_dir / 'test_image.ppm'
            self.assertTrue(expected_ppm.exists(), "PPM file should be created")
            
            # Verify it's a valid PPM file
            with Image.open(expected_ppm) as ppm_img:
                self.assertEqual(ppm_img.format, 'PPM', "Should be PPM format")
                self.assertEqual(ppm_img.mode, 'RGB', "Should be RGB mode")
                self.assertEqual(ppm_img.size, (10, 10), "Should maintain original size")
                
        def test_normalize_different_formats(self):
            """Test normalize with different input formats"""
            # Test with PNG
            png_path = self.test_dir / 'test.png'
            self.test_image.save(png_path, 'PNG')
            normalize_image(str(png_path))
            
            expected_ppm = self.test_dir / 'test.ppm'
            self.assertTrue(expected_ppm.exists())
            
        def test_normalize_nonexistent_file(self):
            """Test normalize with nonexistent file"""
            with self.assertRaises(SystemExit):
                normalize_image(str(self.test_dir / 'nonexistent.jpg'))
                
        def test_normalize_grayscale_conversion(self):
            """Test that grayscale images are converted to RGB"""
            # Create grayscale image
            gray_img = Image.new('L', (5, 5), color=128)
            gray_path = self.test_dir / 'gray.jpg'
            gray_img.save(gray_path)
            
            normalize_image(str(gray_path))
            
            ppm_path = self.test_dir / 'gray.ppm'
            with Image.open(ppm_path) as ppm_img:
                self.assertEqual(ppm_img.mode, 'RGB', "Grayscale should be converted to RGB")
                
        def test_makesample_valid_input(self):
            """Test makesample with valid input"""
            # First create a PPM file
            ppm_path = self.test_dir / 'test.ppm'
            self.test_image.save(ppm_path, format='PPM')
            
            # Test makesample
            json_input = json.dumps({
                "image": str(ppm_path),
                "x": 2,
                "y": 2,
                "w": 5,
                "h": 5,
                "color": "blue"
            })
            
            makesample(json_input)
            
            # Check that samples directory was created
            samples_dir = self.test_dir / 'test.samples'
            self.assertTrue(samples_dir.exists())
            
            # Check that sample file was created
            sample_file = samples_dir / '2_2_5_5.blue'
            self.assertTrue(sample_file.exists())
            
            # Verify the sample is correct size
            with Image.open(sample_file) as sample_img:
                self.assertEqual(sample_img.size, (5, 5))
                self.assertEqual(sample_img.format, 'PPM')
                
        def test_makesample_invalid_json(self):
            """Test makesample with invalid JSON"""
            with self.assertRaises(SystemExit):
                makesample('{"invalid": json}')
                
        def test_makesample_missing_parameters(self):
            """Test makesample with missing parameters"""
            json_input = json.dumps({"image": "test.ppm", "x": 0, "y": 0})
            with self.assertRaises(SystemExit):
                makesample(json_input)
                
        def test_makesample_non_ppm_image(self):
            """Test makesample with non-PPM image"""
            json_input = json.dumps({
                "image": str(self.test_image_path),  # This is a JPG
                "x": 0,
                "y": 0,
                "w": 5,
                "h": 5,
                "color": "red"
            })
            with self.assertRaises(SystemExit):
                makesample(json_input)
                
        def test_makesample_coordinates_out_of_bounds(self):
            """Test makesample with coordinates exceeding image bounds"""
            ppm_path = self.test_dir / 'test.ppm'
            self.test_image.save(ppm_path, format='PPM')
            
            json_input = json.dumps({
                "image": str(ppm_path),
                "x": 8,
                "y": 8,
                "w": 5,  # This would go beyond the 10x10 image
                "h": 5,
                "color": "green"
            })
            with self.assertRaises(SystemExit):
                makesample(json_input)
                
        def test_train_basic_setup(self):
            """Test that train function can be called without crashing on basic setup"""
            # Create PPM files with different colors
            ppm_path1 = self.test_dir / 'sample1.ppm'
            ppm_path2 = self.test_dir / 'sample2.ppm'
            
            # Create test images
            red_img = Image.new('RGB', (50, 50), color='red')
            blue_img = Image.new('RGB', (50, 50), color='blue')
            
            red_img.save(ppm_path1, format='PPM')
            blue_img.save(ppm_path2, format='PPM')
            
            # Create samples directory structure
            samples_dir = self.test_dir / 'test.samples'
            samples_dir.mkdir()
            
            # Copy files with proper naming
            shutil.copy(ppm_path1, samples_dir / '0_0_50_50.red')
            shutil.copy(ppm_path2, samples_dir / '0_0_50_50.blue')
            
            # Test train with minimal iterations
            json_input = json.dumps({
                "samples": str(samples_dir),
                "window": 20,
                "traincount": 2
            })
            
            # This should not crash (we're just testing the setup)
            try:
                train_model(json_input)
            except Exception as e:
                # Allow for missing PyTorch or other dependencies in test environment
                if "No module named" in str(e):
                    self.skipTest(f"Skipping due to missing dependency: {e}")
                else:
                    raise
    
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalyzer)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)
    else:
        print("All tests passed!")

def main():
    parser = argparse.ArgumentParser(description='Analyzer - Image analysis tools')
    parser.add_argument('--normalize', type=str, help='Normalize image to PPM P6 format')
    parser.add_argument('--makesample', type=str, help='Create sample subimage from JSON parameters')
    parser.add_argument('--train', type=str, help='Train neural network on samples from JSON parameters')
    parser.add_argument('--model', type=str, help='Path to existing model file to load/modify')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            run_unit_tests()
        elif args.normalize:
            normalize_image(args.normalize)
        elif args.makesample:
            makesample(args.makesample)
        elif args.train:
            train_model(args.train, args.model)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
