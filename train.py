import numpy as np
import tqdm
import os
import cv2
from models.losses import Losses
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from models.generator import Generator
from models.discriminator import Discriminator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle


class Train:
    def __init__(self, size, epochs, step_per_epoch):
        self.size = size
        self.epochs = epochs
        self.step_per_epoch = step_per_epoch

        self.g_loss = Losses(256).gen_loss
        self.d_loss = Losses(256).disc_loss

        self.g_optimizer = Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.99)
        self.d_optimizer = Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.99)

        self.gen_model = Generator(256).gmodel()
        self.disc_model = Discriminator(256).dmodel()

    def train(self):
        list_photos = os.listdir("dataset/trainA")
        list_cartoons = os.listdir("dataset/trainB")
        list_smooths = os.listdir("dataset/trainB_smooth")

        batchsize_photos = int(len(list_photos) / self.step_per_epoch)
        batchsize_cartoons = int(len(list_cartoons) / self.step_per_epoch)
        batchsize_smooths = int(len(list_smooths) / self.step_per_epoch)

        for i in range(self.epochs):
            epoch = i + 1
            loss_gen = 0
            loss_disc = 0

            pbar = tqdm.tqdm(range(self.step_per_epoch))

            list_photos = shuffle(list_photos)
            list_cartoons, list_smooths = shuffle(list_cartoons, list_smooths)

            def paths2tensor(paths_dir, paths_list):
                tensor = []
                for img in paths_list:
                    path_img = os.path.join(paths_dir, img)
                    x = cv2.imread(path_img)
                    x = cv2.resize(x, (self.size, self.size))
                    tensor.append(x)
                tensor = np.array(tensor)
                tensor = tensor / 255.
                return tensor

            for j in pbar:
                data_list_photos = list_photos[batchsize_photos * j: batchsize_photos * (j + 1)]
                data_list_cartoons = list_cartoons[batchsize_cartoons * j: batchsize_cartoons * (j + 1)]
                data_list_smooths = list_smooths[batchsize_smooths * j: batchsize_smooths * (j + 1)]

                data_photos = paths2tensor("dataset/trainA", data_list_photos)
                data_cartoons = paths2tensor("dataset/trainB", data_list_cartoons)
                data_smooths = paths2tensor("dataset/trainB_smooth", data_list_smooths)

                '''train gen model'''
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # gen
                    output_gen = self.gen_model(data_photos)
                    # Loss gen
                    loss_gen = self.g_loss(data_photos, output_gen)

                    # disc
                    d_gen_photos = self.disc_model(output_gen)
                    d_cartoons = self.disc_model(data_cartoons)
                    d_smooths = self.disc_model(data_smooths)

                    # Loss disc
                    loss_disc = self.d_loss(d_gen_photos, d_cartoons, d_smooths)
                    pbar.set_description("Epoch: {0} || g_loss: {1} || d_loss: {2}".format(epoch, loss_gen, loss_disc))

                    gen_gradients = gen_tape.gradient(loss_gen, self.gen_model.trainable_weights)
                    self.g_optimizer.apply_gradients(zip(gen_gradients, self.gen_model.trainable_weights))

                    disc_gradients = disc_tape.gradient(loss_disc, self.disc_model.trainable_weights)
                    self.d_optimizer.apply_gradients(zip(disc_gradients, self.disc_model.trainable_weights))

            self.gen_model.save_weights('weights/first_save.h5')


if __name__ == "__main__":
    train = Train(256, 1, 6)
    train.train()
