from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, BatchNormalization


class Discriminator(Model):
    def __init__(self, size):
        self.size = size

    def d1_flat_block(self, inp):
        x = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same')(inp)
        x = LeakyReLU()(x)
        return x

    def d2_down_blocks(self, inp):
        x = Conv2D(64, 3, (2, 2), padding='same')(inp)
        x = LeakyReLU()(x)
        x = Conv2D(128, 3, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(128, 3, (2, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 3, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def d1_feature_construct_block(self, inp):
        x = Conv2D(256, 3, (1, 1), padding='same')(inp)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def dmodel(self):
        inp = Input(shape=(self.size, self.size, 3))
        x = self.d1_flat_block(inp)
        x = self.d2_down_blocks(x)
        x = self.d1_feature_construct_block(x)
        out = Conv2D(1, 7, (1, 1), padding='same', activation='sigmoid')(x)
        model = Model(inp, out)
        return model


if __name__ == '__main__':
    disc_model = Discriminator(256).dmodel()
    disc_model.summary()
