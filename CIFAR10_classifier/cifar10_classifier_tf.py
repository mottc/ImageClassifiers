import os

import tensorflow as tf


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
dataset = dataset.repeat()
dataset = dataset.batch(100, drop_remainder=True)

image_holder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
label_holder = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
keep_prob1 = tf.placeholder(tf.float32)


model = tf.layers.conv2d(image_holder, 32, [3, 3], padding='same', activation='relu')
model = tf.layers.conv2d(model, 32, [3, 3],activation='relu')
model = tf.layers.max_pooling2d(model, [2, 2], [1,1],padding='same')
model = tf.layers.dropout(model,rate=keep_prob)


model = tf.layers.conv2d(model, 64, [3, 3], padding='same', activation='relu')
model = tf.layers.conv2d(model, 64, [3, 3], activation='relu')
model = tf.layers.max_pooling2d(model, [2, 2], [1,1])

model = tf.layers.dropout(model,rate=keep_prob)

model = tf.layers.flatten(model)
# reshape = tf.reshape(model, [32, -1])

model = tf.layers.dense(model, 512, activation='relu')
model = tf.layers.dropout(model,rate=keep_prob1)

output = tf.layers.dense(model, 10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label_holder))

# train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
train_op = tf.train.RMSPropOptimizer(0.0001,decay=1e-6).minimize(loss)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label_holder, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(5000):
        data = sess.run(next_element)
        if i % 100 == 0:
            loss_val, train_acc = sess.run([loss, acc], feed_dict={image_holder: data[0], label_holder: data[1],keep_prob:1.0,
                                                                   keep_prob1:1.0})
            print('loss:', loss_val, '------- acc:', train_acc)
        else:
            sess.run(train_op, feed_dict={image_holder: data[0], label_holder: data[1],keep_prob:0.25,keep_prob1:0.5})
