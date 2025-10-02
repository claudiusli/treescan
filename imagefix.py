#!/usr/bin/env python3
"""
Image format normalization tool.

Takes a directory path, creates a new directory with "_fix" suffix,
and converts all images to PNG format with consistent properties.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
from typing import List


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}
    image_files = []
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def convert_image_to_png(image_path: Path, output_path: Path) -> None:
    """Convert image to PNG format with consistent properties."""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None
        
        with Image.open(image_path) as img:
            # Load the image data to prevent lazy loading
            img.load()
            
            # Always convert to RGB to ensure consistent channel ordering
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save as PNG with original dimensions
            output_file = output_path / f"{image_path.stem}.png"
            img.save(output_file, format="PNG")
            print(f"Converted {image_path.name} -> {output_file.name}")
            
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert all images in a directory to PNG format")
    parser.add_argument("directory", help="Source directory containing images")
    
    args = parser.parse_args()
    
    # Validate source directory
    source_dir = Path(args.directory)
    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist")
        sys.exit(1)
    
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        sys.exit(1)
    
    # Create target directory
    target_dir = source_dir.parent / f"{source_dir.name}_fix"
    target_dir.mkdir(exist_ok=True)
    print(f"Created target directory: {target_dir}")
    
    # Get all image files
    image_files = get_image_files(source_dir)
    if not image_files:
        print(f"No image files found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files")
    
    # Convert all images to PNG format
    print(f"\nConverting images to PNG format...")
    for img_path in image_files:
        convert_image_to_png(img_path, target_dir)
    
    print(f"\nProcessing complete. Converted images saved to: {target_dir}")


if __name__ == "__main__":
    main()
