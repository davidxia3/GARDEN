import os
import shutil

def filter_unaugmented(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
   
        if len(filename.split("_")) <=3:
            destination_path = os.path.join(dest_dir, filename)
            shutil.copy2(source_path, destination_path)
            print(f"Copied {filename} to {destination_path}")


source_dir = 'SPINACH/augmented_automatic_segmentation/all_images' 
dest_dir = 'SPINACH/augmented_automatic_segmentation/unaugmented_images'


filter_unaugmented(source_dir, dest_dir)
