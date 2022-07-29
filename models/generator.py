from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Conv2DTranspose, Input


class Generator(Model):
    def __init__(self, size):
        self.size = size

    def g1_flat_block(self, inp):
        x = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(inp)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def g2_down_blocks(self, inp):
        x = Conv2D(128, 3, (2, 2), padding='same')(inp)
        x = Conv2D(128, 3, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, 3, (2, 2), padding='same')(x)
        x = Conv2D(256, 3, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def residual_block(self, inp):
        x = Conv2D(256, 3, (1, 1), padding='same')(inp)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, 3, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = x + inp
        x = ReLU()(x)
        return x

    def g8_residual_blocks(self, num_blocks, inp):
        x = inp
        for i in range(num_blocks):
            x = self.residual_block(x)
        return x

    def g2_up_blocks(self, inp):
        x = Conv2DTranspose(64, 3, (2, 2), padding='same')(inp)
        x = Conv2D(128, 3, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(64, 3, (2, 2), padding='same')(x)
        x = Conv2D(128, 3, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def gmodel(self):
        inp = Input(shape=(self.size, self.size, 3))
        x = self.g1_flat_block(inp)
        x = self.g2_down_blocks(x)
        x = self.g8_residual_blocks(8, x)
        x = self.g2_up_blocks(x)
        out = Conv2D(3, 7, (1, 1), padding='same')(x)
        model = Model(inp, out)
        return model


if __name__ == '__main__':
    gen_model = Generator(256).gmodel()
    gen_model.summary()