import tensorflow as tf

from MNIST_classifier import input_data

max_steps = 3000
MNIST_data_folder = "./MNIST_data"
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)

image_holder = tf.placeholder(tf.float32, shape=[None, 784])
label_holder = tf.placeholder(tf.float32, shape=[None, 10])
drop_prob = tf.placeholder(tf.float32)

model = tf.layers.flatten(image_holder)
model = tf.layers.dense(model, 300, activation='relu')
model = tf.layers.dropout(model, drop_prob)
out = tf.layers.dense(model, 10, activation='softmax')

loss = tf.reduce_mean(-tf.reduce_sum(label_holder * tf.log(out), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(label_holder, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps):
        batch = batch_xs, batch_ys = mnist.train.next_batch(32)
        if i % 100 == 0:
            train_acc = sess.run(acc, feed_dict={image_holder: batch[0], label_holder: batch[1], drop_prob: 0.0})
            print(train_acc)
        else:
            sess.run(train_step, feed_dict={image_holder: batch[0], label_holder: batch[1], drop_prob: 0.25})

    print(
        'test accuracy %g' % sess.run(acc, feed_dict={image_holder: mnist.test.images, label_holder: mnist.test.labels,
                                                      drop_prob: 0.0}))
