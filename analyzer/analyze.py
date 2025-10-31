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
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            run_unit_tests()
        elif args.normalize:
            normalize_image(args.normalize)
        elif args.makesample:
            makesample(args.makesample)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
