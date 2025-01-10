from PIL import Image
import os

content_dir = "data/content/"
resized_dir = "data/content_resized/"
os.makedirs(resized_dir, exist_ok=True)

for filename in os.listdir(content_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(os.path.join(content_dir, filename))
        img_resized = img.resize((640, 360))  # Resize to 512x512
        img_resized.save(os.path.join(resized_dir, filename))
print("Content images resized and saved!")
