#!/usr/bin/env python3

import argparse
import json
import sys
import os
from pathlib import Path
from PIL import Image

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

def main():
    parser = argparse.ArgumentParser(description='Analyzer - Image analysis tools')
    parser.add_argument('--normalize', type=str, help='Normalize image to PPM P6 format')
    
    args = parser.parse_args()
    
    try:
        if args.normalize:
            normalize_image(args.normalize)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
