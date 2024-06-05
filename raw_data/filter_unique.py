import os
import shutil
import hashlib

def calculate_image_hash(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return hashlib.sha256(image_data).hexdigest()

def filter_unique(src_directory, dst_directory):
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    image_hashes = {}
    for filename in os.listdir(src_directory):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(src_directory, filename)
            try:
                image_hash = calculate_image_hash(image_path)
                if image_hash not in image_hashes:
                    image_hashes[image_hash] = image_path
            except:
                pass

    for image_path in image_hashes.values():
        shutil.copy(image_path, dst_directory)

src_directory = 'data/diseased_images'
dst_directory = 'data/unique_diseased_images'
filter_unique(src_directory, dst_directory)
