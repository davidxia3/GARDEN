import os
import pandas as pd
import shutil

roi_dir = "classification_preprocessing/roi"
images = os.listdir(roi_dir)


labels = pd.read_csv("classification_preprocessing/labels.csv")
for filename in images:
    filename_without_suffix = filename
    if filename_without_suffix in labels["filename"].values:
        destination = os.path.join('classification_preprocessing/filtered_roi', filename)
        shutil.copy(os.path.join(roi_dir, filename), destination)

    