import tensorflow as tf
from keras.layers import *
from keras.models import Model


class ConvNeXt_Block(Layer):

    def __init__(self, dim, drop_path=0., layer_scale_vale=0.):
        super(ConvNeXt_Block, self).__init__()
        self.dwconv = DepthwiseConv2D(kernel_size=7, padding='same')
        self.norm = LayerNormalization(epsilon=1e-6)

        self.pwconv1 = Dense(4 * dim)
        self.act = Activation('gelu')
        self.pwconv2 = Dense(dim)
        self.drop_path = DropPath(drop_path)
        self.dim = dim
        self.layer_scale_vale = layer_scale_vale

    def build(self, input_shape):
        self.gamma = tf.Variable(
            initial_value=self.layer_scale_vale * tf.ones((self.dim)),
            trainable=True,
            name='gamma'
        )
        self.built = True

    def call(self, x, **kwargs):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x


class Downsample_Block(Layer):
    def __init__(self, dim):
        super().__init__()
        self.LN = LayerNormalization(epsilon=1e-6)
        self.conv = Conv2D(dim, kernel_size=2, strides=2)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = self.LN(inputs)
        x = self.conv(x)
        return x


class DropPath(Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None, **kwargs):
        return self._drop_path(x, self.drop_prob, training)

    def _drop_path(self, inputs, drop_prob, is_training):
        if (not is_training) or (drop_prob == 0):
            return inputs
        keep_prob = 1.0 - drop_prob
        random_tensor = keep_prob
        shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
        random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)

        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output


def creat_convnext_model(input_shape=(256, 256, 3), depths=None, dims=None, drop_path=0., num_features=128,
                         layer_scale_vale=1e-6):
    if dims is None:
        dims = [96, 192, 384, 768]
    if depths is None:
        depths = [3, 3, 9, 3]
    assert (len(depths) >= 1 and (len(dims) == len(depths)))
    assert len(input_shape) == 3

    input = Input(shape=input_shape)

    y = Conv2D(dims[0], kernel_size=4, strides=4)(input)
    y = LayerNormalization(epsilon=1e-6)(y)
    for _ in range(depths[0]):
        y = ConvNeXt_Block(dims[0], drop_path=drop_path, layer_scale_vale=layer_scale_vale)(y)

    for i in range(1, len(dims)):
        y = Downsample_Block(dims[i])(y)
        for _ in range(depths[i]):
            y = ConvNeXt_Block(dims[i], drop_path=drop_path, layer_scale_vale=layer_scale_vale)(y)

    y = GlobalAveragePooling2D()(y)
    y = LayerNormalization(epsilon=1e-6)(y)

    y = Dense(num_features)(y)
    return Model(input, y)


if __name__ == '__main__':
    creat_convnext_model().summary()