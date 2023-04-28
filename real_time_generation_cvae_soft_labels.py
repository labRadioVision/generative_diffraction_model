import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from conditional_vae_with_diffraction import Conditional_VAE
from utils_diffraction_model import generate_conditioned_attenuations_full, generate_conditioned_attenuation_sample, generate_conditioned_attenuation_sample_random
from keras.layers import Lambda, Input, Dense
from keras.models import Model
# from tensorflow.keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse
import os
import math
import scipy.io as sio
import warnings
# import im
#  full generation for varying target dimensions
# not working, generating all samples to 0
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("-latent_dim", default=16, help="set the number of latent dimensions for reconstruction", type=int)
parser.add_argument("-beta", default=0.05, help="set the beta weighting for KL divergence", type=float)
parser.add_argument("-random_height_dim", default=2, help="set to 1 for random height and dimension generation, 0 for assigned target size, but variable positions in a grid, 2 for assigned target size, and position", type=int)
parser.add_argument("-height", default=1.9, help="set the target height", type=float)
parser.add_argument("-tm", default=0.54, help="set the target trasversal max size ", type=float)
parser.add_argument("-H1", default=2.0, help="set the MAX target height", type=float)
parser.add_argument("-T1", default=0.55, help="set the MAX target trasversal max size ", type=float)
parser.add_argument("-H2", default=1.7, help="set the MIN target height", type=float)
parser.add_argument("-T2", default=0.45, help="set the MIN target trasversal max size ", type=float)
parser.add_argument("-pos_x", default=0.5, help="set the target position in x domain (along the los)", type=float)
parser.add_argument("-pos_y", default=0.0, help="set the target position in y domain (across the los)", type=float)
args = parser.parse_args()

random_height_dim = args.random_height_dim
batch_size = 128
num_channels = 1
########### COMBINATIONS IN TRAINING #####
# 1) 200cm height, 55cm trasversal size
# 2) 200cm height, 45 trasversal size
# 3) 170cm height, 55 trasversal size
# 4) 170cm height, 45 trasversal size
num_positions = 75
num_dim = 2
num_heights = 2
num_dimensions = num_dim * num_heights
num_labels = num_positions + num_dim + num_heights
num_classes = num_positions * num_dim * num_heights # total number of cases generated
latent_dim = args.latent_dim
beta = args.beta
# height = 170
# tm = 45
epochs = 500
rotation_points = 101

rotations = np.linspace(-math.pi / 2, 0, rotation_points)
checkpoint_path = "training_cvae_EM_{}_{}_heights_2_dimensions2_soft_labels/cvae.ckpt".format(latent_dim, beta)

if __name__ == '__main__':

    model = Conditional_VAE(latent_dim, num_labels, rotation_points)
    # filename = 'EM_norm_parameters_{}.mat'.format(num_classes)
    filename = 'EM_norm_parameters_{}_numheights_{}_numdim_{}.mat'.format(num_classes, num_heights, num_dim)
    matfile = sio.loadmat(file_name=filename)
    dataset_max = float(matfile['RSS_max'])
    dataset_mean = float(matfile['RSS_mean'])
    dataset_std = float(matfile['RSS_std'])
    model.load_weights(checkpoint_path)
    deviation = 0.12
    step_distance = 0.2
    filename2 = 'nominal_positions.mat'
    matfile = sio.loadmat(file_name=filename2)
    nominal_positions = np.asarray(matfile['nominal_positions'])


    if random_height_dim == 1:
        ########### generate positions with an assigned target dimension
        generation = 250
        att_responses = np.zeros((201, generation, num_positions))
        height = args.height
        tm = args.tm
        for pos in range(num_positions):
            position = [nominal_positions[pos,0],nominal_positions[pos,1]]
            att_responses[:, :, pos] = generate_conditioned_attenuation_sample(model, dataset_max, dataset_mean, dataset_std, rotation_points, rotations, num_labels, num_dimensions, height, tm, nominal_positions, position, deviation, step_distance, generation)

        dict_1 = {"generated_interpolated_attenuations": att_responses, "rotations": rotations,
                  "generation": generation,
                  "height": height, "trasversal_size": tm, "nominal_positions": nominal_positions,
                  "deviation": deviation, "step_distance": step_distance}

        sio.savemat(
            "results/generated_samples_cvae_latent_vars_{}_beta_{}_height_{}_targetdim_{}.mat".format(latent_dim, beta,
                                                                                                      height, tm), dict_1)
    elif random_height_dim == 2:
        ########### generate positions with an assigned target dimension and position
        generation = 250
        att_responses = np.zeros((201, generation, 1))
        height = args.height
        tm = args.tm
        position = [args.pos_x, args.pos_y]
        deviation = 0 # disable position deviation around nominal location
        att_responses[:, :, 0] = generate_conditioned_attenuation_sample(model, dataset_max, dataset_mean,
                                                                           dataset_std, rotation_points, rotations,
                                                                           num_labels, num_dimensions, height, tm,
                                                                           nominal_positions, position, deviation,
                                                                           step_distance, generation)
        dict_1 = {"generated_interpolated_attenuations": att_responses, "rotations": rotations,
                  "generation": generation,
                  "height": height, "trasversal_size": tm, "nominal_positions": nominal_positions,
                  "deviation": deviation, "step_distance": step_distance, "position": position}

        sio.savemat(
            "results/generated_samples_cvae_latent_vars_{}_beta_{}_height_{}_targetdim_{}_position{}.mat".format(latent_dim, beta,
                                                                                                      height, tm, position), dict_1)

    elif random_height_dim == 0:
        # generate all positions for variable target dimensions but limited to min and max values
        generation = 250 * 4
        att_responses = np.zeros((201, generation, num_positions))
        H1 = args.H1
        H2 = args.H2
        T1 = args.T1
        T2 = args.T2
        for pos in range(num_positions):
            att_responses[:, :, pos] = generate_conditioned_attenuation_sample_random(model, dataset_max, dataset_mean, dataset_std, rotation_points, rotations, num_labels, num_dimensions, nominal_positions, pos, deviation, step_distance, T1, T2, H1, H2, generation)
        dict_1 = {"generated_interpolated_attenuations": att_responses, "rotations": rotations,
                  "generation": generation,
                  "height_max": H1, "trasversal_size_max": T1, "height_min": H2, "trasversal_size_min": T2, "nominal_positions": nominal_positions,
                  "deviation": deviation, "step_distance": step_distance}

        sio.savemat(
            "results/generated_samples_cvae_latent_vars_{}_beta_{}_random_heightMAX-MIN_{}-{}_targetdimMAX-MIN_{}-{}.mat".format(latent_dim, beta,
                                                                                                      H1,H2,T1,T2), dict_1)

    # loaded_model = tf.keras.models.load_model('cond_vae_RSS_model')
