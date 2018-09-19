import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras, expand_dims
import os

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [32, 32])
    return image_resized, label


filenames = os.listdir('D:/GitHub/ImageClassifiers/CIFAR10_classifier/train')
lables = []
newnames = []
for filename in filenames:
    temp_list = [0] * 10
    temp_list[int(filename.split('_')[0])] = 1
    lables.append(temp_list)
    newnames.append('D:/GitHub/ImageClassifiers/CIFAR10_classifier/train/'+filename)

filenames = tf.constant(newnames)
labels = tf.constant(lables)

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=50000)
dataset = dataset.batch(32)
dataset = dataset.repeat()

model = keras.Sequential()
model.add(
    keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(keras.layers.MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='same'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(384, activation='relu'))
model.add(keras.layers.Dense(192, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

# 训练模型，以 32 个样本为一个 batch 进行迭代
model.fit(dataset, epochs=1,steps_per_epoch=50000)

model.save_weights('./cifar10_model')
model.load_weights('./cifar10_model')

img = cv2.imread('D:/GitHub/ImageClassifiers/CIFAR10_classifier/test/0_10.jpg')

cv2.imshow('IMG', img)
img = expand_dims(img, 0)
# data1 = np.random.random((1, 32, 32, 3))
print(np.argmax(model.predict(img, steps=1)))
