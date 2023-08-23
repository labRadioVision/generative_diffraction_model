from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, Conv1D, Conv1DTranspose, Flatten, Concatenate, Reshape, Conv2DTranspose, LocallyConnected1D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
class Conditional_VAE(Model):
    def __init__(self,latent_dim, num_classes, rotation_points):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.rotation_points = rotation_points
        # self.encoder_block1 = self.get_conditional_encoder1(latent_dim, num_classes)
        # self.encoder_block2 = self.get_conditional_encoder2(latent_dim=latent_dim,input_size=2304) # to be revised
        self.decoder_block = self.get_conditional_decoder(latent_dim, num_classes)

    # processes input image and flattens feature maps

    # classical vae decoder
    def get_conditional_decoder(self, latent_dim, num_classes):
        z = Input(shape=(latent_dim + num_classes,))
        # x = Dense(units=self.rotation_points + 6, activation='relu')(z)
        # x = Reshape(target_shape=(self.rotation_points + 6,1))(x)
        # q = LocallyConnected1D(filters=64, kernel_size=3, strides=1, activation='relu')(x) # padding='same',
        # r = LocallyConnected1D(filters=32, kernel_size=3, strides=1, activation='relu')(q)
        # decoded_img = LocallyConnected1D(filters=1, kernel_size=3, strides=1)(r)

        x = Dense(units=self.rotation_points, activation='relu')(z)
        # x = Dense(units=self.rotation_points * 2, activation='relu')(x)
        # decoded_img = Dense(units=self.rotation_points, activation='relu')(x)
        # decoded_img = Reshape(target_shape=(self.rotation_points, 1))(decoded_img)
        x = Reshape(target_shape=(self.rotation_points, 1))(x)
        q = Conv1DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(x)  # padding='same',
        r = Conv1DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(q)
        decoded_img = Conv1DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(r)
        # average = Dense(units=1, activation='relu')(decoded_img)
        # stand = Dense(units=1, activation='relu')(decoded_img)
        # print(Model(inputs=z, outputs=[decoded_img]).summary())
        return Model(inputs=z, outputs=[decoded_img])