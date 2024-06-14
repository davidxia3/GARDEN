import os
import hashlib
from collections import defaultdict
from shutil import move

def remove_duplicates(folder):
    hash_dict = defaultdict(list)

    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                hash_dict[file_hash].append(file_path)

    for files_list in hash_dict.values():
        if len(files_list) > 1:
            print(f"Found {len(files_list)} duplicates for '{files_list[0]}':")
            for file_path in files_list[1:]:
                print(f"   Removing duplicate: '{file_path}'")
                os.remove(file_path)


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
    remove_duplicates(folder)
