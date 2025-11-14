#!/usr/bin/env python3

import argparse
import sys
import os

# Add the parent directory to sys.path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from analyzer.image_operations import normalize_image, makesample
from analyzer.model_operations import train_model, test_model, makemask
from analyzer.ui_operations import sampleselector
from analyzer.testing import run_unit_tests


def print_usage():
    """Print usage information"""
    print("Analyzer - Image analysis tools")
    print("\nUsage examples:")
    print("  --normalize <image_file>")
    print("    Example: --normalize 'image.jpg'")
    print()
    print("  --makesample '{\"image\":\"<string>\",\"x\":<int>,\"y\":<int>,\"w\":<int>,\"h\":<int>,\"color\":\"<string>\"}'")
    print("    Example: --makesample '{\"image\":\"test.ppm\",\"x\":10,\"y\":20,\"w\":50,\"h\":50,\"color\":\"blue\"}'")
    print()
    print("  --sampleselector <image_file>")
    print("    Example: --sampleselector 'test.ppm'")
    print("    Interactive tool for selecting samples with mouse")
    print()
    print("  --train '{\"samples\":\"<string>\",\"window\":<int>,\"traincount\":<int>}' [--model <model_file>]")
    print("    Example: --train '{\"samples\":\"test.samples\",\"window\":50,\"traincount\":1000}'")
    print()
    print("  --test '{\"samples\":\"<string>\",\"window\":<int>,\"testcount\":<int>}' --model <model_file>")
    print("    Example: --test '{\"samples\":\"test.samples\",\"window\":50,\"testcount\":100}' --model 50.model")
    print()
    print("  --makemask '{\"image\":\"<string>\"}' --model <model_file>")
    print("    Example: --makemask '{\"image\":\"test.ppm\"}' --model 50.model")
    print()
    print("  --unittest")
    print("    Run unit tests")

def main():
    parser = argparse.ArgumentParser(description='Analyzer - Image analysis tools')
    parser.add_argument('--normalize', type=str, help='Normalize image to PPM P6 format')
    parser.add_argument('--makesample', type=str, help='Create sample subimage from JSON parameters')
    parser.add_argument('--sampleselector', type=str, help='Interactive sample selection tool')
    parser.add_argument('--train', type=str, help='Train neural network on samples from JSON parameters')
    parser.add_argument('--test', type=str, help='Test neural network model on samples from JSON parameters')
    parser.add_argument('--makemask', type=str, help='Create mask bitmap from JSON parameters')
    parser.add_argument('--model', type=str, help='Path to existing model file to load/modify')
    parser.add_argument('--unittest', action='store_true', help='Run unit tests')
    
    args = parser.parse_args()
    
    try:
        if args.unittest:
            run_unit_tests()
        elif args.normalize:
            normalize_image(args.normalize)
        elif args.makesample:
            makesample(args.makesample)
        elif args.sampleselector:
            sampleselector(args.sampleselector)
        elif args.train:
            train_model(args.train, args.model)
        elif args.test:
            test_model(args.test, args.model)
        elif args.makemask:
            makemask(args.makemask, args.model)
        else:
            print_usage()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
