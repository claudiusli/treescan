import sys
from PIL import Image
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <image_file>")
    sys.exit(1)

file_path = sys.argv[1]
img = Image.open(file_path)

plt.imshow(img)
plt.axis("off")  # Hide axes
plt.show()
