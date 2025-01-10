from PIL import Image
import os

style_image = Image.open("data/style/Van_Gogh.jpg")
style_image_resized = style_image.resize((800, 450))
style_image_resized.save("data/style/Van_Gogh_resized.jpg")
print("Style image resized and saved!")
