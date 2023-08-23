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
        self.encoder_block1 = self.get_conditional_encoder1(latent_dim, num_classes)
        # self.encoder_block2 = self.get_conditional_encoder2(latent_dim=latent_dim,input_size=2304) # to be revised
        self.decoder_block = self.get_conditional_decoder(latent_dim, num_classes)

    # processes input image and flattens feature maps
    def get_conditional_encoder1(self, latent_dim, num_classes):
        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), input_shape=(28,28,1)))
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Conv2D(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2))))
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Flatten())
        inputsA = Input(shape=(self.rotation_points,1))
        inputsB = Input(shape=(num_classes,))
        # x = LocallyConnected1D(filters=32, kernel_size=3, strides=1, activation='relu')(inputsA) # Conv1D
        # x1 = LocallyConnected1D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
        x = Conv1D(filters=32, kernel_size=3, strides=1, activation='relu')(inputsA)  # Conv1D
        x1 = Conv1D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
        # inputsB_v2 = Dense(units=num_classes*2, activation='relu')(inputsB)
        # x1 = Dense(units=64, activation='relu')(x)
        x2 = Flatten()(x1)
        # x3 = Model(inputs=inputsA, outputs=x2)
        combined = Concatenate(axis=1)([x2, inputsB])
        # combined = Concatenate(axis=1)([x2, inputsB_v2])
        mu = Dense(units=latent_dim)(combined)
        rho = Dense(units=latent_dim)(combined)
        # mm = tf.keras.Model(inputs=inputs,outputs=[x])
        # ss = Model(inputs=[x3.input, inputsB], outputs=[mu, rho])
        # print(Model(inputs=[inputsA, inputsB], outputs=[mu, rho]).summary())
        return Model(inputs=[inputsA, inputsB], outputs=[mu, rho])

        # return model

    # gets flattened feature maps, and one hot label vector and outputs mu and rho
    # def get_conditional_encoder2(self, latent_dim, input_size):
    #     inputs = tf.keras.Input(shape=(input_size + 10,))  # 10 classes
    #     mu = tf.keras.layers.Dense(units=latent_dim)(inputs)
    #     rho = tf.keras.layers.Dense(units=latent_dim)(inputs)
    #
    #     return tf.keras.Model(inputs=inputs, outputs=[mu, rho])

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

    def call(self,img,labels):
        # encoder q(z|x,y)
        # enc1_output = self.encoder_block1(img, labels)
        # concat feature maps and one hot label vector
        # np.expand_dims(labels, axis=1)
        # img_lbl_concat = np.concatenate((enc1_output,labels),axis=1)
        img = np.reshape(img, (-1, self.rotation_points, 1))
        labels = np.reshape(labels, (-1, self.num_classes, ))
        z_mu,z_rho = self.encoder_block1([img, labels])

        # sampling
        epsilon = tf.random.normal(shape=z_mu.shape,mean=0.0,stddev=1.0)
        z = z_mu + tf.math.softplus(z_rho) * epsilon

        # decoder p(x|z,y)
        z_lbl_concat = np.concatenate((z,labels),axis=1)
        decoded_img = self.decoder_block(z_lbl_concat)

        return z_mu, z_rho, decoded_img

class Conditional_VAE_MIMO(Model):
    def __init__(self,latent_dim, num_classes, rotation_points, antenna_number):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.rotation_points = rotation_points
        self.antenna_number = antenna_number
        self.encoder_block1 = self.get_conditional_encoder1(latent_dim, num_classes)
        # self.encoder_block2 = self.get_conditional_encoder2(latent_dim=latent_dim,input_size=2304) # to be revised
        self.decoder_block = self.get_conditional_decoder(latent_dim, num_classes)

    # processes input image and flattens feature maps
    def get_conditional_encoder1(self, latent_dim, num_classes):
        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), input_shape=(28,28,1)))
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Conv2D(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2))))
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Flatten())
        inputsA_2D = Input(shape=(self.rotation_points,self.antenna_number))
        # inputsA = tf.keras.layers.Flatten(data_format= 'channels_first')(inputsA_2D)
        inputsA_2Db = tf.expand_dims(inputsA_2D, axis=3)
        inputsB = Input(shape=(num_classes,))
        # x = LocallyConnected1D(filters=32, kernel_size=3, strides=1, activation='relu')(inputsA) # Conv1D
        # x1 = LocallyConnected1D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(inputsA_2Db)  # Conv1D v1 32
        x1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x) # v2 64
        # x1b = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu')(x1)  # v2 64
        # x1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(x)
        # inputsB_v2 = Dense(units=num_classes*2, activation='relu')(inputsB)
        # x1 = Dense(units=64, activation='relu')(x)
        x2 = Flatten()(x1)
        # x2 = Flatten()(x1)
        # x3 = Model(inputs=inputsA, outputs=x2)
        combined = Concatenate(axis=1)([x2, inputsB])
        # combined = Concatenate(axis=1)([x2, inputsB_v2])
        mu = Dense(units=latent_dim)(combined)
        rho = Dense(units=latent_dim)(combined)
        # mm = tf.keras.Model(inputs=inputs,outputs=[x])
        # ss = Model(inputs=[x3.input, inputsB], outputs=[mu, rho])
        # print(Model(inputs=[inputsA, inputsB], outputs=[mu, rho]).summary())
        return Model(inputs=[inputsA_2D, inputsB], outputs=[mu, rho])

        # return model

    # gets flattened feature maps, and one hot label vector and outputs mu and rho
    # def get_conditional_encoder2(self, latent_dim, input_size):
    #     inputs = tf.keras.Input(shape=(input_size + 10,))  # 10 classes
    #     mu = tf.keras.layers.Dense(units=latent_dim)(inputs)
    #     rho = tf.keras.layers.Dense(units=latent_dim)(inputs)
    #
    #     return tf.keras.Model(inputs=inputs, outputs=[mu, rho])

    # classical vae decoder
    def get_conditional_decoder(self, latent_dim, num_classes):
        z = Input(shape=(latent_dim + num_classes,))
        # x = Dense(units=self.rotation_points + 6, activation='relu')(z)
        # x = Reshape(target_shape=(self.rotation_points + 6,1))(x)
        # q = LocallyConnected1D(filters=64, kernel_size=3, strides=1, activation='relu')(x) # padding='same',
        # r = LocallyConnected1D(filters=32, kernel_size=3, strides=1, activation='relu')(q)
        # decoded_img = LocallyConnected1D(filters=1, kernel_size=3, strides=1)(r)

        x = Dense(units=self.rotation_points * self.antenna_number, activation='relu')(z)
        # x = Dense(units=self.rotation_points * 2, activation='relu')(x)
        # decoded_img = Dense(units=self.rotation_points, activation='relu')(x)
        # decoded_img = Reshape(target_shape=(self.rotation_points, 1))(decoded_img)
        x = Reshape(target_shape=(self.rotation_points, self.antenna_number))(x)
        x = tf.expand_dims(x, axis=3)
        # qb = Conv2DTranspose(filters=256, kernel_size=3, strides=1, activation='relu', padding='same')(x)
        q = Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(x)  # padding='same',
        r = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(q)
        # r = Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(qb)
        decoded_img = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(r)
        decoded_img = tf.squeeze(decoded_img)
        # average = Dense(units=1, activation='relu')(decoded_img)
        # stand = Dense(units=1, activation='relu')(decoded_img)
        # print(Model(inputs=z, outputs=[decoded_img]).summary())
        return Model(inputs=z, outputs=[decoded_img])

    def call(self,img,labels):
        # encoder q(z|x,y)
        # enc1_output = self.encoder_block1(img, labels)
        # concat feature maps and one hot label vector
        # np.expand_dims(labels, axis=1)
        # img_lbl_concat = np.concatenate((enc1_output,labels),axis=1)
        img = np.reshape(img, (-1, self.rotation_points, self.antenna_number))
        labels = np.reshape(labels, (-1, self.num_classes, ))
        z_mu,z_rho = self.encoder_block1([img, labels])

        # sampling
        epsilon = tf.random.normal(shape=z_mu.shape,mean=0.0,stddev=1.0)
        z = z_mu + tf.math.softplus(z_rho) * epsilon

        # decoder p(x|z,y)
        z_lbl_concat = np.concatenate((z,labels),axis=1)
        decoded_img = self.decoder_block(z_lbl_concat)

        return z_mu, z_rho, decoded_img

class Conditional_VAE_MIMO_IQ(Model):
    def __init__(self,latent_dim, num_classes, rotation_points, antenna_number, IQ_channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.IQ_channels = IQ_channels
        self.rotation_points = rotation_points
        self.antenna_number = antenna_number
        self.encoder_block1 = self.get_conditional_encoder1(latent_dim, num_classes)
        # self.encoder_block2 = self.get_conditional_encoder2(latent_dim=latent_dim,input_size=2304) # to be revised
        self.decoder_block = self.get_conditional_decoder(latent_dim, num_classes)

    # processes input image and flattens feature maps
    def get_conditional_encoder1(self, latent_dim, num_classes):
        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), input_shape=(28,28,1)))
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Conv2D(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2))))
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Flatten())
        inputsA_3D = Input(shape=(self.rotation_points,self.antenna_number, self.IQ_channels))
        # inputsA = tf.keras.layers.Flatten(data_format= 'channels_first')(inputsA_2D)
        # inputsA_3Db = tf.expand_dims(inputsA_3D)
        inputsB = Input(shape=(num_classes,))
        # x = LocallyConnected1D(filters=32, kernel_size=3, strides=1, activation='relu')(inputsA) # Conv1D
        # x1 = LocallyConnected1D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(inputsA_3D)  # Conv1D v1 32
        x1 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x) # v2 64
        x1b = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(x1) # v2 64
        # x1b = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu')(x1)  # v2 64
        # x1 = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(x)
        # inputsB_v2 = Dense(units=num_classes*2, activation='relu')(inputsB)
        # x1 = Dense(units=64, activation='relu')(x)
        x2 = Flatten()(x1b)
        # x2 = Flatten()(x1)
        # x3 = Model(inputs=inputsA, outputs=x2)
        combined = Concatenate(axis=1)([x2, inputsB])
        # combined = Concatenate(axis=1)([x2, inputsB_v2])
        mu = Dense(units=latent_dim)(combined)
        rho = Dense(units=latent_dim)(combined)
        # mm = tf.keras.Model(inputs=inputs,outputs=[x])
        # ss = Model(inputs=[x3.input, inputsB], outputs=[mu, rho])
        # print(Model(inputs=[inputsA_3D, inputsB], outputs=[mu, rho]).summary())
        return Model(inputs=[inputsA_3D, inputsB], outputs=[mu, rho])

        # return model

    # gets flattened feature maps, and one hot label vector and outputs mu and rho
    # def get_conditional_encoder2(self, latent_dim, input_size):
    #     inputs = tf.keras.Input(shape=(input_size + 10,))  # 10 classes
    #     mu = tf.keras.layers.Dense(units=latent_dim)(inputs)
    #     rho = tf.keras.layers.Dense(units=latent_dim)(inputs)
    #
    #     return tf.keras.Model(inputs=inputs, outputs=[mu, rho])

    # classical vae decoder
    def get_conditional_decoder(self, latent_dim, num_classes):
        z = Input(shape=(latent_dim + num_classes,))
        # x = Dense(units=self.rotation_points + 6, activation='relu')(z)
        # x = Reshape(target_shape=(self.rotation_points + 6,1))(x)
        # q = LocallyConnected1D(filters=64, kernel_size=3, strides=1, activation='relu')(x) # padding='same',
        # r = LocallyConnected1D(filters=32, kernel_size=3, strides=1, activation='relu')(q)
        # decoded_img = LocallyConnected1D(filters=1, kernel_size=3, strides=1)(r)

        x = Dense(units=self.rotation_points * self.antenna_number * self.IQ_channels, activation='relu')(z)
        # x = Dense(units=self.rotation_points * 2, activation='relu')(x)
        # decoded_img = Dense(units=self.rotation_points, activation='relu')(x)
        # decoded_img = Reshape(target_shape=(self.rotation_points, 1))(decoded_img)
        x = Reshape(target_shape=(self.rotation_points, self.antenna_number, self.IQ_channels))(x)
        # x = tf.expand_dims(x, axis=3)
        xb = Conv2DTranspose(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')(x)
        q = Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(xb)  # padding='same',
        r = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(q)
        # r = Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(qb)
        decoded_img = Conv2DTranspose(filters=self.IQ_channels, kernel_size=3, strides=1, padding='same')(r)
        # decoded_img = tf.squeeze(decoded_img)
        # average = Dense(units=1, activation='relu')(decoded_img)
        # stand = Dense(units=1, activation='relu')(decoded_img)
        # print(Model(inputs=z, outputs=[decoded_img]).summary())
        return Model(inputs=z, outputs=[decoded_img])

    def call(self,img,labels):
        # encoder q(z|x,y)
        # enc1_output = self.encoder_block1(img, labels)
        # concat feature maps and one hot label vector
        # np.expand_dims(labels, axis=1)
        # img_lbl_concat = np.concatenate((enc1_output,labels),axis=1)
        img = np.reshape(img, (-1, self.rotation_points, self.antenna_number, self.IQ_channels))
        labels = np.reshape(labels, (-1, self.num_classes, ))
        z_mu,z_rho = self.encoder_block1([img, labels])

        # sampling
        epsilon = tf.random.normal(shape=z_mu.shape,mean=0.0,stddev=1.0)
        z = z_mu + tf.math.softplus(z_rho) * epsilon

        # decoder p(x|z,y)
        z_lbl_concat = np.concatenate((z,labels),axis=1)
        decoded_img = self.decoder_block(z_lbl_concat)

        return z_mu, z_rho, decoded_img