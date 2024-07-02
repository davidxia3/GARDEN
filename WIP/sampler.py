import os
import random
import shutil

input_dir = 'WIP/processed_images'
output_dir = 'WIP/sampled_images'
num_samples = 1000

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg'))]

if len(image_files) < num_samples:
    raise ValueError(f"Not enough images in the directory to sample {num_samples} images")

sampled_files = random.sample(image_files, num_samples)

for filename in sampled_files:
    src_path = os.path.join(input_dir, filename)
    dest_path = os.path.join(output_dir, filename)
    shutil.copy(src_path, dest_path)

print(f"Copied {num_samples} images to {output_dir}")
