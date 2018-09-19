import os
import tensorflow as tf
import cv2

save_dir = "./train.tfrecord"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load(img_dir, width, height):
    writer = tf.python_io.TFRecordWriter(save_dir)
    img_list = [imgName for imgName in os.listdir(img_dir)]
    index = 0
    for imgName in img_list:
        index = index + 1
        if index % 1000 == 0:
            print(index)

        img_path = os.path.join(img_dir, imgName)
        label_index = int((imgName.split('_'))[0])

        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        label = [0] * 10
        label[label_index] = 1
        img_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={'image_raw': _bytes_feature(img_raw),
                     'label': _int64_feature(label)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    load('./train', 32, 32)
