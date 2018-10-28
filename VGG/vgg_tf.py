import tensorflow as tf


class VGG16(object):

    def __init__(self, x, keep_prob, num_classes):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        self.create()

    def create(self):
        net = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv1_1')
        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv1_2')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool1')

        net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv2_1')
        net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv2_2')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool2')

        net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv3_1')
        net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv3_2')
        net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv3_3')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool3')

        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv4_1')
        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv4_2')
        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv4_3')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool4')

        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv5_1')
        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv5_2')
        net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                               name='conv5_3')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool5')

        net = tf.layers.flatten(net, name='flatten')
        net = tf.layers.dense(inputs=net, units=4096, activation='relu', name='fc6')
        net = tf.layers.dropout(inputs=net, rate=self.KEEP_PROB, name='fc6_drop')
        net = tf.layers.dense(inputs=net, units=4096, activation='relu', name='fc7')
        net = tf.layers.dropout(inputs=net, rate=self.KEEP_PROB, name='fc7_drop')
        self.logits = tf.layers.dense(inputs=net, units=self.NUM_CLASSES, name='fc8')


X = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name="input")
Y = tf.placeholder(dtype=tf.float32, shape=(None, 1000), name="GT")
model = VGG16(X, 0.5, 1000)
logits = model.logits

loss = tf.losses.softmax_cross_entropy(Y, logits)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    tf.summary.FileWriter("./", sess.graph)
    sess.run(init)
    for i in range(100):
        # TODO:补充输入网络的值
        sess.run(optimizer, feed_dict={X: "", Y: ""})
