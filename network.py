import tensorflow as tf


def double_conv_block_down(initializer, x):
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    return x


def single_conv_block_down(initializer, x):
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    return x


def last_conv_block_down(x, initializer):
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def double_conv_block_up(initializer, skip, x, filters_first=96, filters_second=96):
    x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="nearest")(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2D(filters_first, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters_second, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def single_conv_block_up(initializer, skip, x):
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2D(48, 3, strides=1, padding='same',
                                   kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def double_conv_block_down_3D(initializer, x):
    x = tf.keras.layers.Conv3D(16, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv3D(16, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    return x


def single_conv_block_down_3D(initializer, x):
    x = tf.keras.layers.Conv3D(16, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    return x


def last_conv_block_down_3D(x, initializer):
    x = tf.keras.layers.Conv3D(16, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)
    x = tf.keras.layers.Conv3D(16, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def double_conv_block_up_3D(initializer, skip, x, filters_first=32, filters_second=32):
    x = tf.keras.layers.UpSampling3D(
            size=(2, 2, 2) )(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv3D(filters_first, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv3D(filters_second, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def single_conv_block_up_3D(initializer, skip, x):
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv3D(16, 3, strides=1, padding='same',
                                   kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

###########################


def autoencoder_3D():
    inputs = tf.keras.layers.Input(shape=[None, None, None, 1])
    initializer = tf.keras.initializers.HeNormal(seed=1)
    skips = [inputs]


    #down sample:                                   input: 96^3
    x = double_conv_block_down_3D(initializer, inputs) #48^3
    skips.append(x)
    for _ in range(3):
        x = single_conv_block_down_3D(initializer, x)  #24^3 -> 12^3 ->6^3
        skips.append(x)                             # 3^3
    x = last_conv_block_down_3D(x, initializer)
    for _ in range(4):
        x = double_conv_block_up_3D(initializer, skips.pop(), x)
    x = double_conv_block_up_3D(initializer, skips.pop(), x, 16, 8)
    x = tf.keras.layers.Conv3D(1, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == '__main__':
    x = tf.random.normal([1, 96, 96, 96, 1])
    a = autoencoder_3D()
    print(a(x))
