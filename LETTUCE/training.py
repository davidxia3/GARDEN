import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import itertools
import os
import cv2

SIZE = 256
INPUT_SHAPE = (SIZE, SIZE, 3)

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

def create_html(df, output_file):
    html_content = '<html><head><style>td { vertical-align: top; }</style></head><body><table border="1">'
    html_content += '<tr><th>Version</th><th>Num Layers</th><th>Dropout</th><th>Optimizer</th><th>Batch Size</th><th>Epoch</th><th>Loss</th><th>Accuracy</th><th>Loss Graph</th><th>Accuracy Graph</th><th>Confusion Matrix</th><th>ROC Curve</th></tr>'

    for _, row in df.iterrows():
        loss_graph_html = f'<img src="data:image/png;base64,{row["loss_graph"]}" width="256" height="256"/>'
        accuracy_graph_html = f'<img src="data:image/png;base64,{row["accuracy_graph"]}" width="256" height="256"/>'
        confusion_matrix_html = f'<img src="data:image/png;base64,{row["confusion_matrix"]}" width="256" height="256"/>'
        roc_curve_html = f'<img src="data:image/png;base64,{row["ROC_curve"]}" width="256" height="256"/>'

        html_content += f'<tr><td>{row["version"]}</td>'
        html_content += f'<td>{row["num_layers"]}</td>'
        html_content += f'<td>{row["dropout"]}</td>'
        html_content += f'<td>{row["optimizer"]}</td>'
        html_content += f'<td>{row["batch_size"]}</td>'
        html_content += f'<td>{row["epoch"]}</td>'
        html_content += f'<td>{row["loss"]}</td>'
        html_content += f'<td>{row["accuracy"]}</td>'
        html_content += f'<td>{loss_graph_html}</td>'
        html_content += f'<td>{accuracy_graph_html}</td>'
        html_content += f'<td>{confusion_matrix_html}</td>'
        html_content += f'<td>{roc_curve_html}</td>'

    html_content += '</table></body></html>'

    with open(output_file, 'w') as file:
        file.write(html_content)

data = []
columns = ['version', 'num_layers', 'dropout', 'optimizer', 'batch_size', 'epoch', 'loss', 'accuracy', 'loss_graph', 'accuracy_graph', 'confusion_matrix', 'ROC_curve']

image_directories = ['SPINACH/unaugmented_automatic_segmentation/cutouts/v1u_cutouts', 'SPINACH/unaugmented_automatic_segmentation/cutouts/v2u_cutouts', 'SPINACH/unaugmented_automatic_segmentation/cutouts/v3u_cutouts']
layers = [2]
dropouts = [0.75]
optimizers = ['adam']
batch_sizes = [128]
epochs = [15]

combinations = itertools.product(image_directories, layers, dropouts, optimizers, batch_sizes, epochs)


counter = 0
for combination in combinations:
    print(counter)
    counter = counter + 1
    image_directory = combination[0]
    version = image_directory.split("/")[-1].split("_")[0]
    num_layers = combination[1]
    dropout = combination[2]
    opt = combination[3]
    bs = combination[4]
    ep = combination[5]

    dataset = []
    label = []

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

    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=8)
    X_train = X_train / 255.
    X_test = X_test / 255.

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(num_layers):
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=bs, verbose=1, epochs=ep, validation_data=(X_test, y_test), shuffle=False)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(1, len(loss) + 1)
    plt.plot(epoch_range, loss, 'y', label='Training loss')
    plt.plot(epoch_range, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_graph = plot_to_base64()
    plt.clf()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epoch_range, acc, 'y', label='Training acc')
    plt.plot(epoch_range, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_graph = plot_to_base64()
    plt.clf()

    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = (model.predict(X_test) >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    confusion_matrix_graph = plot_to_base64()
    plt.clf()

    y_preds = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_preds)
    plt.plot([0, 1], [0, 1], 'y--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    roc_curve_graph = plot_to_base64()
    plt.clf()

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    ideal_roc_thresh = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    ideal = ideal_roc_thresh['thresholds'].iloc[0]

    model_data = [version, num_layers, dropout, opt, bs, ep, loss, accuracy, loss_graph, accuracy_graph, confusion_matrix_graph, roc_curve_graph]
    data.append(model_data)

df = pd.DataFrame(data, columns=columns)
create_html(df, 'LETTUCE/unaugmented_models/performance/performance_7u.html')
