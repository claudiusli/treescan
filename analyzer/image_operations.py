import json
import sys
from pathlib import Path
from PIL import Image

def parse_json_string(json_str):
    """Parse JSON string with error handling"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Expected format: {get_expected_format('makesample')}")
        sys.exit(1)

def validate_required_params(params, required_keys, operation_name):
    """Validate that all required parameters are present"""
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        print(f"Missing required parameters for {operation_name}: {missing_keys}")
        print(f"Expected format: {get_expected_format(operation_name)}")
        sys.exit(1)

def parse_json_params(json_str, required_keys, operation_name):
    """Parse and validate JSON parameters"""
    params = parse_json_string(json_str)
    validate_required_params(params, required_keys, operation_name)
    return params

def get_expected_format(operation_name):
    """Return expected JSON format for each operation"""
    formats = {
        'makesample': '{"image":"<string>","x":<integer>,"y":<integer>,"w":<integer>,"h":<integer>,"color":"<string>"}',
        'train': '{"samples":"<string>","window":<integer>,"traincount":<integer>}',
        'test': '{"samples":"<string>","window":<integer>,"testcount":<integer>}',
        'makemask': '{"image":"<string>"}'
    }
    return formats.get(operation_name, "Check documentation")

def convert_image_to_rgb(img):
    """Convert image to RGB mode if needed"""
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

def handle_image_load_error(error, image_path):
    """Handle image loading errors consistently"""
    if "cannot identify image file" in str(error).lower():
        raise ValueError(f"Cannot read image file: {image_path}")
    raise error

def load_ppm_image(image_path, require_ppm=True):
    """Load image with consistent error handling and optional PPM validation"""
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        with Image.open(image_path) as img:
            if require_ppm and img.format != 'PPM':
                raise ValueError(f"Image must be in PPM format, got {img.format}")
            
            img = convert_image_to_rgb(img)
            img.load()  # Prevent lazy loading
            return img
    except Exception as e:
        handle_image_load_error(e, image_path)

def validate_image_bounds(img, x, y, w, h):
    """Validate that coordinates are within image bounds"""
    if x < 0 or y < 0 or x + w > img.width or y + h > img.height:
        raise ValueError(f"Coordinates ({x}, {y}, {w}, {h}) exceed image bounds ({img.width}, {img.height})")

def create_samples_directory(image_path):
    """Create samples directory for the image"""
    input_path = Path(image_path)
    samples_dir = input_path.parent / (input_path.stem + '.samples')
    samples_dir.mkdir(exist_ok=True)
    return samples_dir

def extract_and_save_sample(img, x, y, w, h, color, samples_dir):
    """Extract subimage and save as sample"""
    subimage = img.crop((x, y, x + w, y + h))
    sample_filename = f"{x}_{y}_{w}_{h}.{color}"
    sample_path = samples_dir / sample_filename
    subimage.save(sample_path, format='PPM')
    return sample_path

def normalize_image(image_path):
    """Convert an image to PPM P6 format maintaining original color depth"""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None
        
        # Open the image
        with Image.open(image_path) as img:
            img = convert_image_to_rgb(img)
            
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
    params = parse_json_params(json_str, ['image', 'x', 'y', 'w', 'h', 'color'], 'makesample')
    
    image_path = params['image']
    x, y, w, h = params['x'], params['y'], params['w'], params['h']
    color = params['color']
    
    # Load and validate image
    img = load_ppm_image(image_path, require_ppm=True)
    validate_image_bounds(img, x, y, w, h)
    
    # Create samples directory and save sample
    samples_dir = create_samples_directory(image_path)
    sample_path = extract_and_save_sample(img, x, y, w, h, color, samples_dir)
    
    print(f"Sample saved to: {sample_path}")
