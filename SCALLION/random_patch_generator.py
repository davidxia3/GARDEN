import os
import random
from PIL import Image

input_dir = 'SCALLION/raw_images'
output_dir = 'SCALLION/processed_images'
num_patches_per_image = 100
patch_size = 1024
downscale_size = 256

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_random_patches(image, num_patches, patch_size=1024):
    width, height = image.size
    patches = []

    for _ in range(num_patches):
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        box = (left, top, left + patch_size, top + patch_size)
        patch = image.crop(box)
        patches.append(patch)
    
    return patches

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png','jpg'))]

for filename in image_files:
    with Image.open(os.path.join(input_dir, filename)) as img:
        patches = create_random_patches(img, num_patches_per_image)
        
        base_filename = os.path.splitext(filename)[0]
        
        for idx, patch in enumerate(patches):
            patch = patch.resize((downscale_size, downscale_size), Image.Resampling.LANCZOS)
            patch_filename = f"{base_filename}_patch_{idx + 1}.png"
            patch.save(os.path.join(output_dir, patch_filename))

print(f"Created patches for {len(image_files)} images in {output_dir}")
