from tensorflow import keras
from MNIST_classifier import input_data
import numpy as np

MNIST_data_folder = "./MNIST_data"
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)

model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(10, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(mnist.train.images, mnist.train.labels, epochs=20, batch_size=100)
score = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=100)
print(score)

batch_xs, batch_ys = mnist.test.next_batch(32)
res = model.predict(batch_xs, batch_size=32)
print(np.argmax(batch_ys, axis=-1))
print(np.argmax(res, axis=-1))
