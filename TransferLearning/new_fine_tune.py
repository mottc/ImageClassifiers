import tensorflow as tf
from nets import vgg
from tensorflow.contrib import slim

image_size = 224

tf.reset_default_graph()

images = tf.placeholder(tf.float32, (None, image_size, image_size, 3))
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(images, is_training=False)
with tf.variable_scope("finetune_layers"):
    # 为获取tensord name，先在tensorboard中将图显示出来，然后找到所需的tensor的name。
    newLayer = tf.get_default_graph().get_tensor_by_name("vgg_16/pool5/MaxPool:0")
    net = tf.layers.flatten(newLayer, "flatten")
    net = tf.layers.dense(net, 100)
    net = tf.layers.dense(net, 20)
    out = tf.layers.dense(net, 10, activation="softmax")

y_label = tf.placeholder(tf.int32, (None, 10))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=out)
train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="finetune_layers")
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=train_var)


# 获取需要恢复的变量列表的函数
def get_restore_var_list():
    all_restore_var = []
    # 不使用tf.trainable_variables()，因为batchnorm的moving_mean/variance不属于可训练变量
    for var in tf.global_variables():
        if 'vgg_16/conv' in var.name:
            all_restore_var.append(var)
    return all_restore_var


restore_var_list = get_restore_var_list()
saver = tf.train.Saver(var_list=restore_var_list)
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./tensorboard/new_fine_tune', sess.graph)
    sess.run(tf.variables_initializer(var_list=train_var))
    saver.restore(sess, "./pre_trained_models/vgg_16.ckpt")
