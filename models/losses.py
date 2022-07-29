import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from models.generator import Generator
from models.discriminator import Discriminator
from tensorflow.keras.losses import mean_absolute_error, binary_crossentropy
from tensorflow import reduce_mean
from tensorflow.keras.layers import Input


class Losses:
    def __init__(self, size):
        self.size = size

    def vgg_features(self, x):
        vgg_model = VGG19(input_shape=(self.size, self.size, 3), weights='imagenet', include_top=False)
        layer_conv44 = vgg_model.get_layer("block4_conv4")
        vgg_features = Model(vgg_model.input, layer_conv44.output)
        vgg_features.trainable = False
        return vgg_features(x)

    def content_loss(self, y_true, y_pred):
        content_photos = self.vgg_features(y_true)
        content_gens = self.vgg_features(y_pred)
        loss = reduce_mean(mean_absolute_error(content_photos, content_gens))
        return loss

    def gen_loss(self, y_true, y_pred):
        label_gens = np.ones(shape=y_pred.shape, dtype=float)
        label_loss = reduce_mean(binary_crossentropy(label_gens, y_pred))
        content_loss = self.content_loss(y_true, y_pred)

        gen_loss = label_loss + content_loss * 10
        return gen_loss

    def disc_loss(self, d_gen_photos, d_cartoons, d_smooths):
        label_gens = np.zeros(shape=d_gen_photos.shape, dtype=float)
        label_cartoons = np.ones(shape=d_cartoons.shape, dtype=float)
        label_smooths = np.zeros(shape=d_smooths.shape, dtype=float)

        label_gens_loss = reduce_mean(binary_crossentropy(label_gens, d_gen_photos))
        label_cartoons_loss = reduce_mean(binary_crossentropy(label_cartoons, d_cartoons))
        label_smooths_loss = reduce_mean(binary_crossentropy(label_smooths, d_smooths))

        dis_loss = label_gens_loss + label_cartoons_loss + label_smooths_loss
        return dis_loss






if __name__ == '__main__':
    loss = Losses(256)
