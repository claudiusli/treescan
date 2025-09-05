import sys
from PIL import Image
import matplotlib.pyplot as plt

def show_image(img: Image.Image) -> None:
    """Display the image using matplotlib"""
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    img = Image.open(file_path)
    show_image(img)

if __name__ == "__main__":
    main()
