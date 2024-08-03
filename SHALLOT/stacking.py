import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os
import cv2
from scipy.stats import mode

decision_tree_model_path = 'SHALLOT/models/decision_tree.pkl'
random_forest_model_path = 'SHALLOT/models/forest.pkl'
logistic_regression_model_path = 'SHALLOT/models/logistic.pkl'
deep_learning_model_path = 'LETTUCE/unaugmented_models/model_1.h5'

class_0_dir = 'SHALLOT/healthy_images'
class_1_dir = 'SHALLOT/diseased_images'

image_size = (256, 256) 
batch_size = 32

def load_images_from_directory(directory, label, image_size, convert_to_grayscale=False):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if convert_to_grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0  
            images.append(img)
            labels.append(label)
    return images, labels

images_class_0_gray, labels_class_0_gray = load_images_from_directory(class_0_dir, 0, image_size, convert_to_grayscale=True)
images_class_1_gray, labels_class_1_gray = load_images_from_directory(class_1_dir, 1, image_size, convert_to_grayscale=True)

images_class_0_rgb, labels_class_0_rgb = load_images_from_directory(class_0_dir, 0, image_size, convert_to_grayscale=False)
images_class_1_rgb, labels_class_1_rgb = load_images_from_directory(class_1_dir, 1, image_size, convert_to_grayscale=False)

X_test_gray = np.array(images_class_0_gray + images_class_1_gray)
y_test = np.array(labels_class_0_gray + labels_class_1_gray)

if len(X_test_gray.shape) == 3:
    X_test_gray = np.expand_dims(X_test_gray, axis=-1)

X_test_flat = X_test_gray.reshape(X_test_gray.shape[0], -1)

print(f"X_test_flat shape: {X_test_flat.shape}")

with open(decision_tree_model_path, 'rb') as file:
    decision_tree_model = pickle.load(file)

with open(random_forest_model_path, 'rb') as file:
    random_forest_model = pickle.load(file)

with open(logistic_regression_model_path, 'rb') as file:
    logistic_regression_model = pickle.load(file)

deep_learning_model = load_model(deep_learning_model_path)

dt_predictions = decision_tree_model.predict(X_test_flat)
rf_predictions = random_forest_model.predict(X_test_flat)
lr_predictions = logistic_regression_model.predict(X_test_flat)

X_test_rgb = np.array(images_class_0_rgb + images_class_1_rgb)
dl_predictions = np.argmax(deep_learning_model.predict(X_test_rgb), axis=-1)

dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)
dl_accuracy = accuracy_score(y_test, dl_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Deep Learning Model Accuracy: {dl_accuracy:.4f}")

combined_predictions = np.array([dt_predictions, rf_predictions, lr_predictions, dl_predictions])
voted_predictions = mode(combined_predictions, axis=0).mode[0]

print(f"Voted predictions shape: {voted_predictions.shape}")
print(f"Voted predictions: {voted_predictions}")

if voted_predictions.size > 0:
    ensemble_accuracy = accuracy_score(y_test, voted_predictions)

    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
else:
    print("Voted predictions are empty.")
