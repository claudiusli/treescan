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
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            run_unit_tests()
        elif args.normalize:
            normalize_image(args.normalize)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
