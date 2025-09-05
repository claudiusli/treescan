from PIL import Image

def analyze_image(img: Image.Image) -> None:
    """Analyze the image snippet and print its dimensions"""
    width, height = img.size
    print(f"Dimensions: {width}x{height}")
