import unittest
import tempfile
import shutil
import sys
from pathlib import Path
from PIL import Image
import json
from .image_operations import normalize_image, makesample
from .model_operations import train_model

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
            # Create a JPEG image specifically for this test
            jpeg_path = self.test_dir / 'test.jpg'
            self.test_image.save(jpeg_path, format='JPEG')
            
            json_input = json.dumps({
                "image": str(jpeg_path),  # This is a JPG
                "x": 0,
                "y": 0,
                "w": 5,
                "h": 5,
                "color": "red"
            })
            
            # This should raise ValueError, not SystemExit
            with self.assertRaises(ValueError) as context:
                makesample(json_input)
            
            self.assertIn("Image must be in PPM format", str(context.exception))
                
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
            
            # This should raise ValueError, not SystemExit
            with self.assertRaises(ValueError) as context:
                makesample(json_input)
            
            self.assertIn("exceed image bounds", str(context.exception))
                
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
