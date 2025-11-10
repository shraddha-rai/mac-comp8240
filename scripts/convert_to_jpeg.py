from PIL import Image
import os

folder_path = "new_data/opinion/ensemble"

for filename in os.listdir(folder_path):
    if filename.startswith('.'):                # this is just because .DS_Store in mac kept on throwing error
        continue
    if filename.lower().endswith('.jpeg'):
        continue
    img_path = os.path.join(folder_path, filename)
    img = Image.open(img_path).convert("RGB")
    base_name = os.path.splitext(filename)[0]
    img.save(os.path.join(folder_path, f"{base_name}.jpeg"), "JPEG")
    os.remove(img_path)                             # removing the older image

