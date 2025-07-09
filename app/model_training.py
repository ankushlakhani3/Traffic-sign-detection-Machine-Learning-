import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app import app

data = []
labels = []
classes = 43
cur_path = os.getcwd()
dataset_path = os.path.join(cur_path, 'static/dataset')
classe_labels = app.config['CLASS_LABELS']


def reading_all_images():
    global data
    global labels
    # Retrieving the images and their labels
    for i in range(classes):
        path = os.path.join(dataset_path, 'Train', str(i))
        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(os.path.join(path, a))
                image = image.resize((30, 30))
                image = np.array(image)
                # sim = Image.fromarray(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print("Error loading image")

    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)

    # Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                        random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Converting the labels into one hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    return X_train, X_test, y_train, y_test


def build_model(X_train):
    # Building the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    return model


def compile_and_train_model(model, X_train, y_train, X_test, y_test):
    epochs = 15
    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    # sekf.model.save("my_model.h5")
    return history


def plot_model_accuracy(history):
    # plotting graphs for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def run_test_dataset_on_model(trained_model, model_path):
    # testing accuracy on test dataset
    path = os.path.join(dataset_path, 'Test.csv')
    y_test = pd.read_csv(path)

    image_labels = y_test["ClassId"].values
    images = y_test["Path"].values

    image_data = []

    for img in images:
        img = os.path.join(dataset_path, img)
        image = Image.open(img)
        image = image.resize((30, 30))
        image_data.append(np.array(image))

    image_test_data = np.array(image_data)

    prediction = trained_model.predict(image_test_data)
    prediction = np.argmax(prediction, axis=-1)

    # Accuracy with the test data

    print(accuracy_score(image_labels, prediction))

    trained_model.save(model_path)


def predict_model(saved_model_path, image_file_path):
    trained_model = load_model(saved_model_path)
    image = Image.open(image_file_path)
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    prediction = trained_model.predict(image[:, :, :, :3])[0]
    predicted_label = classe_labels[np.argmax(prediction, axis=-1) + 1]
    print(predicted_label)
    return predicted_label


if __name__ == "__main__":
    model_save_path = os.path.join(os.getcwd(), 'static', 'traffic_classifier.h5')
    X_train, X_test, y_train, y_test = reading_all_images()
    model = build_model(X_train)
    history = compile_and_train_model(model, X_train, y_train, X_test, y_test)
    # plot_model_accuracy(history)
    run_test_dataset_on_model(model, model_save_path)
    # file_path = os.path.join(dataset_path, 'Meta', '42.png')
    # predict_model(model_save_path, file_path)
