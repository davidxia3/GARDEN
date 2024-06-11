import os

directory = "manual_segmentation/images"

file_list = os.listdir(directory)

for filename in file_list:
    if filename.endswith(".JPG"):
        new_filename = os.path.join(directory, filename[:-4] + ".jpg")
        os.rename(os.path.join(directory, filename), new_filename)