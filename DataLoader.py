import pandas as pd
import numpy as np
import tensorflow as tf
from Options import option
import glob
import os


class DataLoader(option):
    def __init__(self, data_type):
        super(DataLoader, self).__init__()

        Path = f'{self.root}/{data_type}'

        imgs = glob.glob(f'{Path}/images/*.png')
        masks = glob.glob(f'{Path}/labels/*.png')

        # self.label_dict = pd.read_csv(f'{self.root}')

        self.img_ds = tf.data.Dataset.from_tensor_slices(imgs)
        self.mask_ds = tf.data.Dataset.from_tensor_slices(masks)
        # self.cnt = len(self.img_label_path)

    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, 3)
        img = tf.image.resize(images=img, size=[self.H, self.W]) / 255.

        return img

    def decode_mask(self, mask_path):
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_png(mask, 1)
        mask = tf.cast(tf.image.resize(images=mask, size=[self.H, self.W]), dtype=tf.uint8)

        class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        one_hot = []
        for index in class_index:
            class_map = tf.reduce_all(tf.equal(mask, index), axis=-1)
            one_hot.append(class_map)

        one_hot = tf.stack(one_hot, axis=-1)
        one_hot = tf.cast(one_hot, tf.float32)

        return one_hot

    def load_ds(self):
        img_ds = self.img_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        mask_ds = self.mask_ds.map(self.decode_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return tf.data.Dataset.zip((img_ds, mask_ds))


def configure_for_performance(ds, cnt, shuffle=False):
    if shuffle:
        ds = ds.shuffle(buffer_size=cnt)
        ds = ds.batch(1)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    else:
        ds = ds.batch(1)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds