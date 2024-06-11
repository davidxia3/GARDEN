import os
import shutil
import random

# Define paths
images_path = 'manual_segmentation/images'
masks_path = 'manual_segmentation/masks'

# Define output directories
train_images_path = 'automatic_segmentation/train/images'
train_masks_path = 'automatic_segmentation/train/masks'
val_images_path = 'automatic_segmentation/valid/images'
val_masks_path = 'automatic_segmentation/valid/masks'
test_images_path = 'automatic_segmentation/test/images'
test_masks_path = 'automatic_segmentation/test/masks'

# Create output directories if they don't exist
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_masks_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_masks_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_masks_path, exist_ok=True)

# Function to get list of files, filtering out system files
def get_filtered_file_list(path):
    return sorted([f for f in os.listdir(path) if not f.startswith('.')])

# Get list of files
images = get_filtered_file_list(images_path)
masks = get_filtered_file_list(masks_path)

# Debug: print the number of images and masks
print(f"Number of images: {len(images)}")
print(f"Number of masks: {len(masks)}")

# Ensure that the images and masks are paired correctly
if len(images) != len(masks):
    print("Mismatch in the number of images and masks.")
    print("Images without corresponding masks:")
    mask_names = [os.path.splitext(mask)[0] for mask in masks]
    for img in images:
        if os.path.splitext(img)[0] not in mask_names:
            print(f" - {img}")
    print("Masks without corresponding images:")
    image_names = [os.path.splitext(img)[0] for img in images]
    for mask in masks:
        if os.path.splitext(mask)[0] not in image_names:
            print(f" - {mask}")
    raise ValueError("The number of images and masks must be the same")

for img, msk in zip(images, masks):
    if img.split('.')[0] != msk.split('.')[0]:
        print(f"Image and mask prefix mismatch: {img} vs {msk}")
        raise ValueError("Each image and mask must have the same prefix")

# Combine images and masks into pairs
data_pairs = list(zip(images, masks))

# Shuffle the data
random.seed(42)  # For reproducibility
random.shuffle(data_pairs)

# Define split sizes
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Calculate split indices
total_size = len(data_pairs)
train_size = int(total_size * train_split)
val_size = int(total_size * val_split)
test_size = total_size - train_size - val_size

# Split the data
train_pairs = data_pairs[:train_size]
val_pairs = data_pairs[train_size:train_size + val_size]
test_pairs = data_pairs[train_size + val_size:]

# Function to copy files
def copy_files(pairs, src_images_path, src_masks_path, dst_images_path, dst_masks_path):
    for img_file, mask_file in pairs:
        shutil.copy(os.path.join(src_images_path, img_file), os.path.join(dst_images_path, img_file))
        shutil.copy(os.path.join(src_masks_path, mask_file), os.path.join(dst_masks_path, mask_file))

# Copy the files to their respective directories
copy_files(train_pairs, images_path, masks_path, train_images_path, train_masks_path)
copy_files(val_pairs, images_path, masks_path, val_images_path, val_masks_path)
copy_files(test_pairs, images_path, masks_path, test_images_path, test_masks_path)

print("Data has been split into training, validation, and test sets.")
