import os

def rename(folder_path):
    files = sorted(os.listdir(folder_path))  # Sort files to ensure consistent order
    
    # Get the last two components of the path for species and status
    path_components = folder_path.split(os.sep)
    species = path_components[-2] if len(path_components) > 1 else ''
    status = path_components[-1] if len(path_components) > 0 else ''
    
    for index, file_name in enumerate(files):
        new_file_name = f"{species}_{status}_{index}{os.path.splitext(file_name)[1]}"
        
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        
        os.rename(old_file_path, new_file_path)

def get_leaf_folders(root_folder):
    leaf_folders = []
    
    for root, dirs, files in os.walk(root_folder):
        if not dirs:
            leaf_folders.append(root)
    
    return leaf_folders

root_folder = 'raw_data'
leaf_folders = get_leaf_folders(root_folder)
print("Folders with no children folders:")
for folder in leaf_folders:
    print(folder)
    rename(folder)
