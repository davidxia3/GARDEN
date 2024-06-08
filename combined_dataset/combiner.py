import os
import shutil

def copy_jpg_files(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for root, _, files in os.walk(src_folder):
        for file_name in files:
            if file_name.endswith((".jpg", ".JPG")):
                full_file_name = os.path.join(root, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dst_folder)
                    print(f"Copied: {full_file_name} to {dst_folder}")

src_folder = 'preprocessing'
dst_folder = 'combined_dataset/images'
copy_jpg_files(src_folder, dst_folder)
src_folder = 'augmentations'
copy_jpg_files(src_folder, dst_folder)

