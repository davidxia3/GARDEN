from PIL import Image, ImageOps
import os

def add_padding(input_path, output_path, size=256):
    image = Image.open(input_path)
    
    max_side = max(image.size)
    padded_image = ImageOps.pad(image, (max_side, max_side), color='black')
    
    resized_image = padded_image.resize((size, size), Image.LANCZOS)
    
    resized_image.save(output_path)

def get_leaf_folders(root_folder):
    leaf_folders = []
    
    for root, dirs, files in os.walk(root_folder):
        if not dirs:
            leaf_folders.append(root)
    
    return leaf_folders

root_folder = 'SPINACH/raw_data'
leaf_folders = get_leaf_folders(root_folder)
print("Folders with no children folders:")
for folder in leaf_folders:
    print(folder)
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            input_path = os.path.join(folder, file)
            
            relative_path = os.path.relpath(input_path, root_folder)
                    
            output_path = os.path.join('SPINACH/preprocessing', relative_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            add_padding(input_path, output_path, 256)
