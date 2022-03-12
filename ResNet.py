from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
from methods import GroupBlock


def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def Group_BN_Relu(filters, kernel_size, strides, input_layer):
    input_layer = Conv2D(input_layer.shape[-1], kernel_size, strides=strides, padding='same')(input_layer)
    q = Reshape((-1, filters))(Conv2D(filters, 1, strides=1, padding='same')(input_layer))
    k = Reshape((-1, filters))(Conv2D(filters, 1, strides=1, padding='same')(input_layer))
    k = Lambda(lambda x: tf.transpose(x, (0, 2, 1)))(k)
    adj = tf.matmul(q, k)
    x = GroupBlock.My_gcn(filters)([Reshape((-1, input_layer.shape[-1]))(input_layer), adj])
    x = Reshape((input_layer.shape[1], input_layer.shape[2], -1))(x)
    return x


def resiidual_c_or_d(input_x, filters, flag):
    if flag == 'c':
        # 主路
        x = Conv_BN_Relu(filters, (1, 1), 1, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)
        x = Conv_BN_Relu(filters * 4, (1, 1), 1, x)

        # 输出
        y = Add()([x, input_x])

        return y
    elif flag == 'd':
        # 主路
        x = Conv_BN_Relu(filters, (1, 1), 2, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)
        x = Conv_BN_Relu(filters * 4, (1, 1), 1, x)

        # 支路下采样
        input_x = Conv_BN_Relu(filters * 4, (1, 1), 2, input_x)

        # 输出
        y = Add()([x, input_x])

        return y


def build():

    input_layer = Input((224, 224, 3))
    conv1 = Group_BN_Relu(64, (7, 7), 1, input_layer)
    conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)
    x = conv1_Maxpooling

    # 中间层
    filters = 64
    num_residuals = [3, 4, 23, 3]
    for i, num_residual in enumerate(num_residuals):
        for j in range(num_residual):
            if j == 0:
                x = resiidual_c_or_d(x, filters, 'd')
            else:
                x = resiidual_c_or_d(x, filters, 'c')
        filters = filters * 2

    # 最后一层
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(2)(x)
    x = Dropout(0.5)(x)
    y = Softmax(axis=-1)(x)

    model = Model([input_layer], [y])
    return model


if __name__ == '__main__':

    build().summary()

