import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=8)

class_weight_options = [
    {0: 1, 1: 1},   
    {0: 1, 1: 0.8}, 
    {0: 1, 1: 0.6}, 
    {0: 1, 1: 0.4},
    {0: 1, 1: 0.2},  
]

clf = DecisionTreeClassifier(random_state=8)

param_grid = {'class_weight': class_weight_options}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='f1')

grid_search.fit(X_train, y_train)

print(f'Best class weights: {grid_search.best_params_}')
print(f'Best F1 score: {grid_search.best_score_}')

best_clf = DecisionTreeClassifier(random_state=8, class_weight=grid_search.best_params_['class_weight'])
best_clf.fit(X_train, y_train)

with open('SHALLOT/models/decision_tree.pkl', 'wb') as file:
    pickle.dump(best_clf, file)

y_pred = best_clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)

y_prob = best_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig('roc_curve.png')
plt.close()

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
