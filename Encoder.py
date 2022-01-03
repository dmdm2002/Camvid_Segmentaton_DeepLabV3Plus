from tensorflow.keras.layers import Conv2D, Concatenate, BatchNormalization, ReLU, UpSampling2D, AveragePooling2D


class ASPP(object):
    def __init__(self, x, filters, num_class):
        super(ASPP, self).__init__()
        self.filters = filters
        self.num_class = num_class
        self.x = x

    def conv_1by1(self, x):
        x = Conv2D(filters=self.filters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def conv_3by3(self, x, dilation_rate):
        x = Conv2D(filters=self.filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                   dilation_rate=dilation_rate, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def build(self):
        input_shape = self.x.get_shape().as_list()
        _, h, w, f = input_shape

        branch1 = self.conv_1by1(self.x)
        branch2 = self.conv_3by3(self.x, dilation_rate=6)
        branch3 = self.conv_3by3(self.x, dilation_rate=12)
        branch4 = self.conv_3by3(self.x, dilation_rate=18)

        out_img = AveragePooling2D(pool_size=(h, w))(self.x)
        out_img = self.conv_1by1(out_img)
        out_img = UpSampling2D(size=(h, w), interpolation='bilinear')(out_img)

        concat = Concatenate()([branch1, branch2, branch3, branch4, out_img])
        concat = Conv2D(filters=(self.filters*5), kernel_size=(1, 1))(concat)
        concat = ReLU()(concat)

        return concat