from keras.layers import *
from keras.models import Model, Sequential
import numpy as np
import keras.backend as K


def AdaptiveAvgPool2d(x, outsize):
    x_shape = K.int_shape(x)
    batchsize1, dim1, dim2, channels1 = x_shape
    stride = np.floor(dim1 / outsize).astype(np.int32)
    kernels = dim1 - (outsize - 1) * stride
    adpooling = AveragePooling2D(pool_size=(kernels, kernels), strides=(stride, stride))(x)

    return adpooling


def conv_3x3_bn(inp, filters, image_size, downsample=False):
    strides = 1 if downsample is False else 2

    def call(x):
        x = Conv2D(filters, 3, strides, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        return x

    return call


class PreNorm(Layer):
    def __init__(self, fn, norm, block_type='C'):
        super(PreNorm, self).__init__()
        self.norm = norm
        self.fn = fn
        self.block = block_type
        assert self.block == 'C' or 'T'

    def call(self, inputs, *args, **kwargs):
        if self.block == 'C':
            return self.fn(self.norm(inputs))
        else:
            return self.fn(self.norm(inputs), self.norm(inputs))


class SE(Layer):
    def __init__(self, inp, oup, expansion=0.25):
        super(SE, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d
        self.fc = Sequential(
            [
                Dense(int(inp * expansion), use_bias=False),
                Activation('gelu'),
                Dense(oup, use_bias=False),
                Activation('sigmoid')
            ]
        )

    def call(self, inputs, *args, **kwargs):
        y = self.avg_pool(inputs, 1)
        assert K.int_shape(y)[1] == 1 and K.int_shape(y)[2] == 1
        y = y[:, 0, 0]
        y = self.fc(y)
        y = Reshape((1, 1, -1))(y)
        return inputs * y


class MBConv(Layer):

    def __init__(self, inp, oup, image_size, downsample=False, expansion=4, block_type='C'):
        super(MBConv, self).__init__()
        self.downsample = downsample
        self.block_type = block_type
        strides = 1 if self.downsample is False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = MaxPooling2D(strides=2, padding='same')
            self.proj = Conv2D(oup, 1, 1, padding='valid', use_bias=False)

        if expansion == 1:
            self.conv = Sequential(
                [
                    Conv2D(hidden_dim, 3, strides, padding='same', use_bias=False),
                    BatchNormalization(),
                    Activation('gelu'),
                    Conv2D(oup, 1, 1, use_bias=False),
                    BatchNormalization()
                ]
            )
        else:
            self.conv = Sequential(
                [
                    Conv2D(hidden_dim, 1, strides, padding='valid', use_bias=False),
                    BatchNormalization(),
                    Activation('gelu'),

                    Conv2D(hidden_dim, 3, 1, padding='same', use_bias=False,
                           groups=hidden_dim),
                    BatchNormalization(),
                    Activation('gelu'),

                    SE(inp, hidden_dim),
                    #
                    Conv2D(oup, 1, 1, use_bias=False),
                    BatchNormalization()
                ]
            )
            # self.conv = self.Conv(inp, oup, hidden_dim, strides)

        self.conv = PreNorm(self.conv, BatchNormalization(), block_type=self.block_type)

    def call(self, x, *args, **kwargs):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class FeedForward(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = Sequential(
            [
                Dense(hidden_dim),
                Activation('gelu'),
                Dropout(dropout),
                Dense(dim),
                Dropout(dropout)
            ]
        )

    def call(self, inputs, *args, **kwargs):
        return self.net(inputs)


class Transformer(Layer):
    def __init__(self, inp, oup, image_size, downsample=False, block_type='T',
                 dropout=0., heads=8, key_dim=32):
        super(Transformer, self).__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample
        self.block_type=block_type
        if self.downsample:
            self.pool1 = MaxPooling2D()
            self.pool2 = MaxPooling2D()
            self.proj = Conv2D(oup, 1, 1, use_bias=False)

        # self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = Sequential(
            [
                Reshape((self.ih * self.iw, -1)),
                PreNorm(self.attn, LayerNormalization(), block_type=self.block_type),
                Dense(oup, use_bias=False),
                Reshape((self.ih, self.iw, -1))
            ]
        )
        self.ff = Sequential(
            [
                Reshape((self.ih * self.iw, -1)),
                PreNorm(self.ff, LayerNormalization(), block_type=self.block_type),
                Dense(oup, use_bias=False),
                Reshape((self.ih, self.iw, -1))
            ]
        )

    def call(self, inputs, *args, **kwargs):
        if self.downsample:
            x = self.proj(self.pool1(inputs)) + self.attn(self.pool2(inputs))
        else:
            x = inputs + self.attn(inputs)
        x = x + self.ff(x)
        return x


class CoAtNet:
    def __init__(self, image_size, num_blocks, channels, num_classes=2,
                 block_types=['C', 'C', 'T', 'T']):
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(conv_3x3_bn, 3, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = AveragePooling2D()
        self.fc = Dense(num_classes, use_bias=False)

    def forward(self, inputs):
        x = self.s0(inputs)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        #
        x = self.pool(x)
        x = Flatten()(x)
        x = self.fc(x)
        return Model(inputs, x)

    def _make_layer(self, block, inp, oup, depth, image_size):
        def call(x):
            for i in range(depth):
                if i == 0:
                    x = block(inp, oup, image_size=image_size, downsample=True)(x)

                else:
                    x = block(inp, oup, image_size=image_size)(x)
            return x

        return call


def coatnet_0():
    inputs = Input(shape=(224, 224, 3))
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((K.int_shape(inputs)[1], K.int_shape(inputs)[2]),
                   num_blocks, channels, num_classes=2).forward(inputs)


def coatnet_1():
    inputs = Input(shape=(224, 224, 3))
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((K.int_shape(inputs)[1], K.int_shape(inputs)[2]),
                   num_blocks, channels, num_classes=2).forward(inputs)


def coatnet_2():
    inputs = Input(shape=(224, 224, 3))
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [128, 128, 256, 512, 1026]  # D
    return CoAtNet((K.int_shape(inputs)[1], K.int_shape(inputs)[2]),
                   num_blocks, channels, num_classes=2).forward(inputs)


def coatnet_3():
    inputs = Input(shape=(224, 224, 3))
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CoAtNet((K.int_shape(inputs)[1], K.int_shape(inputs)[2]),
                   num_blocks, channels, num_classes=2).forward(inputs)


if __name__ == '__main__':
    net = coatnet_0()
    net.summary()
