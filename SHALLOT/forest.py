import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

class1_dir = 'SHALLOT/healthy_images'
class2_dir = 'SHALLOT/diseased_images'

img_size = (256, 256)

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  
        img = img.resize(img_size) 
        img_array = np.array(img).flatten()  
        images.append(img_array)
        labels.append(label)
    return images, labels

class1_images, class1_labels = load_images_from_folder(class1_dir, 0)  
class2_images, class2_labels = load_images_from_folder(class2_dir, 1)  

X = np.array(class1_images + class2_images)
y = np.array(class1_labels + class2_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

class_weight_options = [
    {0: 1, 1: 1},  
    {0: 1, 1: 0.9}, 
    {0: 1, 1: 0.8}, 
    {0: 1, 1: 0.7},   
    {0: 1, 1: 0.6}, 
    {0: 1, 1: 0.5},  
    {0: 1, 1: 0.4},
    {0: 1, 1: 0.3}, 
    {0: 1, 1: 0.2},  
    {0: 1, 1: 0.1}, 
]

rf = RandomForestClassifier(random_state=8)

param_grid = {'class_weight': class_weight_options}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1')

grid_search.fit(X_train, y_train)

print(f'Best class weights: {grid_search.best_params_}')
print(f'Best F1 score: {grid_search.best_score_}')

best_rf = RandomForestClassifier(random_state=8, class_weight=grid_search.best_params_['class_weight'])
best_rf.fit(X_train, y_train)

with open('SHALLOT/models/forest.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(cr)
