import os
import json
import csv

image_dir = "manual_segmentation/images"
mask_dir = "manual_segmentation/masks"

images = os.listdir(image_dir)
images = sorted(images)

masks = os.listdir(mask_dir)
masks = sorted(masks)

label_list = []
filenames_list = []

for filename in masks:
    filenames_list.append(filename[0:-5] + ".png")
    with open(os.path.join(mask_dir, filename), "r") as file:
        data = json.load(file)
        label_list.append(data["shapes"][0]["label"])


csv_file_path = 'classification_preprocessing/labels.csv'

with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['filename', 'label'])
    
    for name, age in zip(filenames_list, label_list):
        writer.writerow([name, age])

