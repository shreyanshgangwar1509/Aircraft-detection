import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import pathlib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


data_dir = pathlib.Path('aircrafts/crop')

data = list(data_dir.glob('*'))  # Correct pattern

#folders = {folder.name for folder in data if folder.is_dir()}
#print(folders)
folders = [folder.name for folder in data_dir.glob('*') if folder.is_dir()]


aircraft_label_dict = {folder: idx for idx, folder in enumerate(folders)}


#print(aircraft_label_dict)
aircraft_image_dict = {folder.name: list(folder.glob('*')) for folder in data_dir.glob('*') if folder.is_dir()}

first_folder = folders[0] 
first_image_path = aircraft_image_dict[first_folder][0] 

img = PIL.Image.open(str(first_image_path))
img.show()

print(first_image_path) 
aircraft_image_dict = {folder.name: list(folder.glob('*')) for folder in data_dir.glob('*') if folder.is_dir()}


#print(aircraft_image_dict)
# img_train = cv2.imread(aircraft_image_dict['A10'][0])
# img_train = img_train/255;
x,y = [],[]

for aircraft_name,images in aircraft_image_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        x.append(resized_img)
        y.append(aircraft_label_dict[aircraft_name])

x = np.array(x)
y = np.array(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
x_train = x_train/255
x_test = x_test/255
data_argument = Sequential([
    layers.RandomFlip('horizontal',input_shape=(180,180,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])

num_class = 77
cnn = Sequential([
    data_argument,
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(120,activation='relu'),
    layers.Dense(num_class)
])

cnn.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy
            (from_logits=True),metrics=['accuracy'])

cnn.fit(x_train,y_train,epochs=5)

cnn.evaluate(x_test,y_test)

pred = cnn.predict(x_test)
score = tf.nn.softmax(pred[0])
np.argmax(score)
