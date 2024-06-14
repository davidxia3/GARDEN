import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.metrics import auc
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

image_directory = 'automatic_segmentation/cutouts/v1_cutouts'
SIZE = 256
dataset = []   
label = []

features = []

images = os.listdir(image_directory)
for i, image_name in enumerate(images):

    image = cv2.imread(os.path.join(image_directory, image_name))

    dataset.append(image)

    l = image_name.split("_")[2]
    if l == "healthy":
        label.append(0)
    else:
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 8)
X_train = X_train / 255.
X_test = X_test / 255.

INPUT_SHAPE = (SIZE, SIZE, 3)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.75))

model.add(Dense(1))
model.add(Activation('sigmoid'))  

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',     #rmsprop, adam  
              metrics=['accuracy'])

print(model.summary())   

history = model.fit(X_train, 
                         y_train, 
                         batch_size = 128, 
                         verbose = 1, 
                         epochs = 5,      
                         validation_data=(X_test,y_test),
                         shuffle = False
                     )


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()

print(type(loss))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.clf()


_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

mythreshold=0.5
y_pred = (model.predict(X_test)>= mythreshold).astype(int)
cm=confusion_matrix(y_test, y_pred)  
sns.heatmap(cm, annot=True)
plt.show()

plt.clf()

y_preds = model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
     
plt.clf()
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
print("Ideal threshold is: ", ideal_roc_thresh['thresholds']) 

auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)
