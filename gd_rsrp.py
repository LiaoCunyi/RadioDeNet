import os
import tensorflow as tf
import numpy as np
from tensorflow import keras


class Downsample(keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = keras.layers.Conv2D(filters,
                                         (size, size),
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(keras.Model):

    def __init__(self, filters, size, apply_dropout=True):
        super(Upsample, self).__init__()

        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = keras.layers.Conv2DTranspose(filters,
                                                    (size, size),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
        self.batchnorm = keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = keras.layers.Dropout(0.5)

    def call(self, x1, x2, training=None):

        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x

class ConvBlock1(keras.Model):
    def __init__(self, filters, size, apply_dropout=True):
        super(ConvBlock1, self).__init__()

        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv = self.conv1 = keras.layers.Conv2D(filters,
                                         (size, size),
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        self.batchnorm = keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = keras.layers.Dropout(0.5)

    def call(self, x1, training=None):

        x = self.conv(x1)
        x = tf.nn.relu(x)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)

        return x

class ConvBlock2(keras.Model):
    def __init__(self, filters, size, apply_dropout=True):
        super(ConvBlock2, self).__init__()

        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1  = keras.layers.Conv2D(filters,
                                         (size, size),
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        self.conv2  = keras.layers.Conv2D(filters,
                                         (size, size),
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        self.batchnorm = keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = keras.layers.Dropout(0.5)

    def call(self, x1, training=None):

        x = self.conv1(x1)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)

        return x

class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)
        #
        # self.CNN = keras.layers.Conv2D(1, (4, 4),
        #                                  strides=1,
        #                                  padding='same',
        #                                  kernel_initializer=initializer,
        #                                  use_bias=False)

        self.down1 = Downsample(32, 4, apply_batchnorm=False)
        self.down2 = Downsample(64, 4)
        self.down3 = Downsample(128, 4)
        self.down4 = Downsample(256, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)


        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(256, 4, apply_dropout=True)
        self.up3 = Upsample(128, 4)
        self.up4 = Upsample(64, 4)
        self.up5 = Upsample(32, 4)
        self.convblock1 = ConvBlock1(4, 1)
        self.convblock2 = ConvBlock2(1, 1)
        self.convblock3 = ConvBlock2(1, 1)

        self.last = keras.layers.Conv2DTranspose(1, (4, 4),
                                                 strides=2,
                                                 padding='same',
                                                 kernel_initializer=initializer)

    def call(self, inputs,training=None):
        # x shape == (bs, 64, 64, 2)
        # x = self.CNN(x, training=training)
        x, y = inputs

        x = self.convblock1(x, training=training)
        x1 = self.down1(x, training=training)  # (bs, 32, 32, 16)  (bs, 16, 16, 16)
        x2 = self.down2(x1, training=training)  # (bs, 16, 16, 32)    (bs, 8, 8, 16)
        x3 = self.down3(x2, training=training)  # (bs, 8, 8, 64)  (bs, 4, 4, 16)
        x4 = self.down4(x3, training=training)  # (bs, 4, 4, 128)  (bs, 2, 2, 16)
        x5 = self.down5(x4, training=training)  # (bs, 2, 2, 128)  (bs, 1, 1, 16)
        x6 = self.down6(x5, training=training)  # (bs, 1, 1, 128)

        u1 = self.up1(x6, x5, training=training)  # (bs, 2, 2, 128)
        u2 = self.up2(u1, x4, training=training)  # (bs, 4, 4, 128)
        u3 = self.up3(u2, x3, training=training)  # (bs, 8, 8, 64)
        u4 = self.up4(u3, x2, training=training)  # (bs, 16, 16, 32)
        u5 = self.up5(u4, x1, training=training)  # (bs, 32, 32, 16)


        # u1 = self.up1(x5, x4, training=training)  # (bs, 2, 2, 128)
        # u2 = self.up2(u1, x3, training=training)  # (bs, 4, 4, 128)
        # u3 = self.up3(u2, x2, training=training)  # (bs, 8, 8, 64)
        # u4 = self.up4(u3, x1, training=training)  # (bs, 16, 16, 32)

        out = self.last(u5)  # (bs, 64, 64, 1)
        # out = self.last(u4)  # (bs, 64, 64, 1)
        u = tf.concat([y, out], axis=-1)
        u = self.convblock2(u,training=training)
        out = self.convblock3(u, training=training)

        out = tf.nn.tanh(out)

        return out


class DiscDownsample(keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)
        self.maxpooling = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

        self.conv1 = keras.layers.Conv2D(filters, (size, size),
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = keras.layers.BatchNormalization()

    def call(self, x, training=None):

        x = self.conv1(x)
        x = self.maxpooling(x)
        x = tf.nn.leaky_relu(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        return x

class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)


        # self.down1 = DiscDownsample(64, 4, False)
        # self.down2 = DiscDownsample(128, 4)
        # self.down3 = DiscDownsample(256, 4)

        self.down1 = DiscDownsample(32, 3, False)
        self.down2 = DiscDownsample(64, 3)
        self.down3 = DiscDownsample(128, 3)

        self.averagepooling = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
        # pool_size 是缩小比例的因数

        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = keras.layers.ZeroPadding2D()
        self.conv = keras.layers.Conv2D(256, (3, 3),
                                        strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)
        self.conv1 = keras.layers.Conv2D(128, (3, 3),
                                        strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.batchnorm2 = keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = keras.layers.ZeroPadding2D()
        self.last = keras.layers.Conv2D(1, (4, 4),
                                        strides=1,
                                        kernel_initializer=initializer)

    def call(self, inputs, training=None):
        inp, target = inputs

        # concatenating the input and the target
        x = tf.concat([inp, target], axis=-1)  # (bs, 64, 64, channels*2)

        x = self.down1(x, training=training)  # (bs, 32, 32, 64)
        x = self.down2(x, training=training)  # (bs, 16, 16, 128)
        x = self.down3(x, training=training)  # (bs, 8, 8, 256)

        x = self.averagepooling(x) # (bs, 4, 4, 256)

        x = self.zero_pad1(x)  # (bs, 6, 6, 256)
        x = self.conv(x)  # (bs, 4, 4, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 6, 6, 256)
        x = self.conv1(x)  # (bs, 4, 4, 512)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.leaky_relu(x)

        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, 8, 8, 1)
        # x = tf.nn.sigmoid(x)

        return x
