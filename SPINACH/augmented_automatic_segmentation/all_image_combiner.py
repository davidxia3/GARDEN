import os
import shutil

source_dirs = ['SPINACH/augmentations', 'SPINACH/preprocessing']

destination_dir = 'SPINACH/augmented_automatic_segmentation/all_images'
os.makedirs(destination_dir, exist_ok=True)


for source_dir in source_dirs:
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                source_path = os.path.join(root, filename)
                
                if not filename.endswith('.jpg'):
                    filename = os.path.splitext(filename)[0] + '.jpg'
                    
                destination_path = os.path.join(destination_dir, filename)
                shutil.copy2(source_path, destination_path)

print(f"Copied all .JPG files to {destination_dir} and renamed to .jpg")
