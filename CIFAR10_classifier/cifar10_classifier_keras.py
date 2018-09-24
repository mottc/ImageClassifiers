import os

import tensorflow as tf
from tensorflow import keras


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [32, 32])
    return image_resized, label


img_names = os.listdir('./train')
labels = []
file_names = []
for img_name in img_names:
    temp_label = [0.0] * 10
    temp_label[int(img_name.split('_')[0])] = 1.0
    labels.append(temp_label)
    file_names.append('./train/' + img_name)

file_names = tf.constant(file_names)
labels = tf.constant(labels)

dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=50000)
dataset = dataset.batch(100, drop_remainder=True)
dataset = dataset.repeat()

model = keras.Sequential()
model.add(
    keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(keras.layers.Conv2D(filters=32, kernel_size=[3, 3], activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=[2, 2]))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=[2, 2]))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='softmax'))

opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(dataset, epochs=10, steps_per_epoch=500)

# model.save_weights('./saved_model/cifar10_model')
#
# model.load_weights('./saved_model/cifar10_model')

# img = cv2.imread('./test/0_10.jpg')
#
# img = expand_dims(img, 0)
# print(np.argmax(model.predict(img, steps=1)))
