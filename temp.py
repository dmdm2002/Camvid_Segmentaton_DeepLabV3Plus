import cv2
import tensorflow as tf

root = 'D:/sideproject/camvid/dataset/train/labels/0001TP_006690.png'

mask = tf.io.read_file(root)
mask = tf.io.decode_png(mask)
class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

one_hot = []
for index in class_index:
    class_map = tf.reduce_all(tf.equal(mask, index), axis=-1)
    one_hot.append(class_map)

one_hot = tf.stack(one_hot, axis=-1)
one_hot = tf.cast(one_hot, tf.float32)

print(one_hot.shape)