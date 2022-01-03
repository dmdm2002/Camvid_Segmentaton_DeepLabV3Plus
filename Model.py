import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, UpSampling2D, Concatenate, Dropout
from tensorflow.keras import applications
from Encoder import ASPP


class DeepLabV3Plus(object):
    def __init__(self, filters, num_class):
        super(DeepLabV3Plus, self).__init__()
        self.filters = filters
        self.num_class = num_class
        self.baseModel = applications.ResNet101(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

    def get_ASPP(self):
        feature = self.baseModel.get_layer('conv4_block23_out').output

        aspp = ASPP(feature, 256, self.num_class)
        aspp_feature = aspp.build()

        return aspp_feature

    def build(self):
        low_level_feature = self.baseModel.get_layer('conv2_block3_out').output
        low_level_feature = Conv2D(48, kernel_size=(1, 1), use_bias=False)(low_level_feature)
        low_level_feature = BatchNormalization()(low_level_feature)
        low_level_feature = ReLU()(low_level_feature)

        input_shape = low_level_feature.get_shape().as_list()
        _, h, w, f = input_shape

        aspp = self.get_ASPP()
        aspp = UpSampling2D(size=(4, 4), interpolation='bilinear')(aspp)

        x = Concatenate()([aspp, low_level_feature])

        x = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), padding='same',
                   kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)

        x = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), padding='same',
                   kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.1)(x)

        x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        x = Conv2D(filters=self.num_class, kernel_size=(1, 1), strides=1)(x)

        return keras.Model(inputs=self.baseModel.input, outputs=x)