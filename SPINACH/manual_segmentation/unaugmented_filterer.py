import os
import shutil

image_dir = 'SPINACH/manual_segmentation/images'
mask_dir = 'SPINACH/manual_segmentation/processed_masks'

image_dest_dir = 'SPINACH/manual_segmentation/unaugmented_images'
mask_dest_dir = 'SPINACH/manual_segmentation/unaugmented_processed_masks'

if not os.path.exists(image_dest_dir):
    os.makedirs(image_dest_dir)

if not os.path.exists(mask_dest_dir):
    os.makedirs(mask_dest_dir)

for filename in os.listdir(image_dir):
    if len(filename.split("_")) <= 3:
        src_file = os.path.join(image_dir, filename)
        dst_file = os.path.join(image_dest_dir, filename)
        shutil.copy(src_file, dst_file)
        print(f"Copied: {src_file} to {dst_file}")

for filename in os.listdir(mask_dir):
    if len(filename.split("_")) <= 3:
        src_file = os.path.join(mask_dir, filename)
        dst_file = os.path.join(mask_dest_dir, filename)
        shutil.copy(src_file, dst_file)
        print(f"Copied: {src_file} to {dst_file}")