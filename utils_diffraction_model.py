import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import scipy.io as sio
import time
import math
warnings.filterwarnings("ignore")

########### COMBINATIONS IN TRAINING #####
# 1) 2m height, 55cm trasversal size
# 2) 2m height, 45 trasversal size
# 3) 1.7 height, 55 trasversal size
# 4) 1.7 height, 45 trasversal size
H1 = 2.0
H2 = 1.7
T1 = 0.55
T2 = 0.45
filename2 = 'nominal_target_size.mat'
matfile = sio.loadmat(file_name=filename2)
nominal_target_size = np.asarray(matfile['nominal_target_size'])

def assignLabelsPositions(position_c, nominal_positions, num_labels, step_distance):
    label_onehot = np.zeros((1, num_labels))
    distance = np.zeros(nominal_positions.shape[0])
    for k in range(nominal_positions.shape[0]):
        distance[k] = math.sqrt(math.pow(position_c[0] - nominal_positions[k, 0],2) + math.pow(position_c[1] - nominal_positions[k, 1],2))
    sd = np.where(distance < step_distance)
    d_control = sd[0].size
    if d_control == 1:
        label_onehot[:, sd] = 1.0
    elif d_control > 1:
        label_onehot[:, sd] = 1 / d_control
        # label_onehot[:, sd] = 1.0
    else:
        print("position outside the training area")
    return label_onehot

def assignLabelsPositions_v2(position_c, nominal_positions, num_labels, step_distance):
    label_onehot = np.zeros((1, num_labels))
    distance = np.zeros(nominal_positions.shape[0])
    for k in range(nominal_positions.shape[0]):
        distance[k] = math.sqrt(math.pow(position_c[0] - nominal_positions[k, 0],2) + math.pow(position_c[1] - nominal_positions[k, 1],2))
    sd = np.where(distance < step_distance)
    d_control = sd[0].size
    if d_control == 1:
        label_onehot[:, sd] = 1.0
    elif d_control > 1:
        # label_onehot[:, sd] = 1 / d_control
        label_onehot[:, sd] = 1.0
    else:
        print("position outside the training area")
    return label_onehot

def assignLabelsPositions_v3(position_c, nominal_positions, num_labels):
    label_onehot = np.zeros((1, num_labels))
    distance = np.zeros(nominal_positions.shape[0])
    for k in range(nominal_positions.shape[0]):
        distance[k] = math.sqrt(math.pow(position_c[0] - nominal_positions[k, 0],2) + math.pow(position_c[1] - nominal_positions[k, 1],2))
    sd = np.argmin(distance)
    label_onehot[:, sd] = 1.0
    return label_onehot

def assignLabelsTargeSize(height, tm):
    if H1 >= height >= H2 and tm <= T1 and tm >= T2:
        coeff_tm = (tm - T2) / (T1 - T2)
        coeff_h = (height - H2) / (H1 - H2)
        # he = [0, 1, 2, 3]
        label_v = [(coeff_h) * (coeff_tm), (coeff_h) * (1.0 - coeff_tm), (1.0 - coeff_h) * (coeff_tm), (1.0 - coeff_h) * (1.0 - coeff_tm)]
    else:
        print("unrecognized target dimension, set to default")
        # he = [0, 1, 2, 3]
        label_v = [1.0, 0.0, 0.0, 0.0]
    return label_v

def assignLabelsTargeSize_v2(height, tm, step_distance2):
    siz = nominal_target_size.shape[0]
    label_v = np.zeros((1, siz))
    distance = np.zeros(siz)
    for k in range(siz):
        distance[k] = math.sqrt(
            math.pow(height - nominal_target_size[k, 0], 2) + math.pow(tm - nominal_target_size[k, 1], 2))
    sd = np.where(distance < step_distance2)
    d_control = sd[0].size
    if d_control == 1:
        label_v[:, sd] = 1.0
    elif d_control > 1:
        label_v[:, sd] = 1.0
        # label_onehot[:, sd] = 1
    else:
        print("target size outside the training area")
    return label_v

def assignLabelsTargeSize_v3(height, tm):
    siz = nominal_target_size.shape[0]
    label_v = np.zeros((1, siz))
    distance = np.zeros(siz)
    for k in range(siz):
        distance[k] = math.sqrt(
            math.pow(height - nominal_target_size[k, 0], 2) + math.pow(tm - nominal_target_size[k, 1], 2))
    sd = np.argmin(distance)
    label_v[:, sd] = 1.0
    return label_v


def generate_conditioned_attenuation_sample_random_binlabels(model,dataset_max,dataset_mean, dataset_std, rotation_points, rotations, num_labels, num_dimensions, nominal_positions, pos, deviation, step_distance, step_distance2, T1_input, T2_input, H1_input, H2_input, generation):
    n = generation  # number of generation per input
    # deviation = 0.12
    # step_distance = 0.2
    # latent = model.encoder_block1.output[0].shape[1]
    num_positions = nominal_positions.shape[0]
    latent = model.latent_dim
    rotations2 = rotations[:-1]
    full_rotations = np.concatenate((rotations, rotations2[::-1]), axis=0)  # -pi/2 pi/2 support
    att_responses = np.zeros((np.shape(full_rotations)[0], n))
    # label_onehot = np.zeros((1, num_labels))


    # labelling (initial)
    # label_onehot[:, pos] = 1.0


    for gen_idx in range(n):
        deviationX = np.random.rand() * (deviation * 2) - deviation # +- 0.1
        deviationY = np.random.rand() * (deviation * 2) - deviation
        position_c = [nominal_positions[pos,0] + deviationX, nominal_positions[pos,1] + deviationY]
        # labelling, choose
        label_onehot = assignLabelsPositions_v2(position_c, nominal_positions, num_labels, step_distance)
        # label_onehot = assignLabelsPositions_v3(position_c, nominal_positions, num_labels)
        ##############
        height = (H1_input - H2_input) * np.random.rand() + H2_input
        tm = (T1_input - T2_input) * np.random.rand() + T2_input
        # labelling, choose
        # label_v = assignLabelsTargeSize_v3(height, tm)
        label_v = assignLabelsTargeSize_v2(height, tm, step_distance2)
        #####################
        label_onehot[:, num_positions:] = label_v
        # for q in range(num_labels - num_positions):
        #    label_onehot[:, num_positions + q] = label_v[:, q]
        z = tf.random.normal(shape=(1, latent), mean=0.0, stddev=1.0)
        z_lbl_concat = np.concatenate((z, label_onehot), axis=1)
        preds = model.decoder_block(z_lbl_concat)

        response = tf.reshape(preds[0], [-1, rotation_points])
        response = (response * dataset_std) + dataset_mean
        response = response * dataset_max
        g = np.squeeze(response.numpy())
        g2 = g[:-1]
        att_responses[:, gen_idx] = np.concatenate((g, g2[::-1]), axis=0)
    return att_responses

def generate_conditioned_attenuation_sample_random(model,dataset_max,dataset_mean, dataset_std, rotation_points, rotations, num_labels, num_dimensions, nominal_positions, pos, deviation, step_distance, T1_input, T2_input, H1_input, H2_input, generation):
    n = generation  # number of generation per input
    # deviation = 0.12
    # step_distance = 0.2
    # latent = model.encoder_block1.output[0].shape[1]
    latent = model.latent_dim
    rotations2 = rotations[:-1]
    full_rotations = np.concatenate((rotations, rotations2[::-1]), axis=0)  # -pi/2 pi/2 support
    att_responses = np.zeros((np.shape(full_rotations)[0], n))
    # label_onehot = np.zeros((1, num_labels))


    # labelling (initial)
    # label_onehot[:, pos] = 1.0


    for gen_idx in range(n):
        deviationX = np.random.rand() * (deviation * 2) - deviation # +- 0.1
        deviationY = np.random.rand() * (deviation * 2) - deviation
        position_c = [nominal_positions[pos,0] + deviationX, nominal_positions[pos,1] + deviationY]
        # labelling
        label_onehot = assignLabelsPositions(position_c, nominal_positions, num_labels, step_distance)
        height = (H1_input - H2_input) * np.random.rand() + H2_input
        tm = (T1_input - T2_input) * np.random.rand() + T2_input
        label_v = assignLabelsTargeSize(height, tm)
        #label_v = assignLabelsTargeSize_v2(height, tm)
        for q in range(num_dimensions):
            label_onehot[:, num_labels - num_dimensions + q] = label_v[q]
        z = tf.random.normal(shape=(1, latent), mean=0.0, stddev=1.0)
        z_lbl_concat = np.concatenate((z, label_onehot), axis=1)
        preds = model.decoder_block(z_lbl_concat)

        response = tf.reshape(preds[0], [-1, rotation_points])
        response = (response * dataset_std) + dataset_mean
        response = response * dataset_max
        g = np.squeeze(response.numpy())
        g2 = g[:-1]
        att_responses[:, gen_idx] = np.concatenate((g, g2[::-1]), axis=0)
    return att_responses

def generate_conditioned_attenuation_sample_binlabels(model,dataset_max,dataset_mean, dataset_std, rotation_points, rotations, num_labels, num_dimensions, height, tm, nominal_positions, position, deviation, step_distance, step_distance2, generation):
    n = generation  # number of generation per input
    num_positions = nominal_positions.shape[0]
    # deviation = 0.12
    # step_distance = 0.2
    # latent = model.encoder_block1.output[0].shape[1]
    latent = model.latent_dim
    rotations2 = rotations[:-1]
    full_rotations = np.concatenate((rotations, rotations2[::-1]), axis=0)  # -pi/2 pi/2 support
    att_responses = np.zeros((np.shape(full_rotations)[0], n))
    # labelling, choose
    # label_v = assignLabelsTargeSize_v3(height, tm)
    label_v = assignLabelsTargeSize_v2(height, tm, step_distance2)
    # ######################

    for gen_idx in range(n):
        deviationX = np.random.rand() * (deviation * 2) - deviation # +- 0.1
        deviationY = np.random.rand() * (deviation * 2) - deviation
        position_c = [position[0] + deviationX, position[1] + deviationY]
        # labelling, choose
        label_onehot = assignLabelsPositions_v2(position_c, nominal_positions, num_labels, step_distance)
        # label_onehot = assignLabelsPositions_v3(position_c, nominal_positions, num_labels)
        ##################################
        label_onehot[:, num_positions:] = label_v
        # for q in range(num_labels - num_positions):
        #    label_onehot[:, num_positions + q] = label_v[:, q]
        z = tf.random.normal(shape=(1, latent), mean=0.0, stddev=1.0)
        z_lbl_concat = np.concatenate((z, label_onehot), axis=1)
        preds = model.decoder_block(z_lbl_concat)

        response = tf.reshape(preds[0], [-1, rotation_points])
        response = (response * dataset_std) + dataset_mean
        response = response * dataset_max
        g = np.squeeze(response.numpy())
        g2 = g[:-1]
        att_responses[:, gen_idx] = np.concatenate((g, g2[::-1]), axis=0)
    return att_responses

def generate_conditioned_attenuation_sample(model,dataset_max,dataset_mean, dataset_std, rotation_points, rotations, num_labels, num_dimensions, height, tm, nominal_positions, position, deviation, step_distance, generation):
    n = generation  # number of generation per input
    # deviation = 0.12
    # step_distance = 0.2
    # latent = model.encoder_block1.output[0].shape[1]
    latent = model.latent_dim
    rotations2 = rotations[:-1]
    full_rotations = np.concatenate((rotations, rotations2[::-1]), axis=0)  # -pi/2 pi/2 support
    att_responses = np.zeros((np.shape(full_rotations)[0], n))
    # label_onehot = np.zeros((1, num_labels))
    label_v = assignLabelsTargeSize(height, tm)
    # label_v = assignLabelsTargeSize_v2(height, tm)
    # labelling (initial)
    # label_onehot[:, pos] = 1.0


    for gen_idx in range(n):
        deviationX = np.random.rand() * (deviation * 2) - deviation # +- 0.1
        deviationY = np.random.rand() * (deviation * 2) - deviation
        position_c = [position[0] + deviationX, position[1] + deviationY]
        # labelling
        label_onehot = assignLabelsPositions(position_c, nominal_positions, num_labels, step_distance)
        for q in range(num_dimensions):
            label_onehot[:, num_labels - num_dimensions + q] = label_v[q]
        z = tf.random.normal(shape=(1, latent), mean=0.0, stddev=1.0)
        z_lbl_concat = np.concatenate((z, label_onehot), axis=1)
        preds = model.decoder_block(z_lbl_concat)

        response = tf.reshape(preds[0], [-1, rotation_points])
        response = (response * dataset_std) + dataset_mean
        response = response * dataset_max
        g = np.squeeze(response.numpy())
        g2 = g[:-1]
        att_responses[:, gen_idx] = np.concatenate((g, g2[::-1]), axis=0)
    return att_responses

def generate_conditioned_attenuations_full(model, dataset_max, dataset_mean, dataset_std, frequency_points, rotations, epoch, num_labels, num_cases, num_positions, num_heights, num_dimensions, beta):
    # full generation i.e., for varying dimensions and heights
    n = 250  # number of generation per input
    # latent = model.encoder_block1.output[0].shape[1]
    latent = model.latent_dim
    rotations2 = rotations[:-1]
    full_rotations = np.concatenate((rotations, rotations2[::-1]), axis=0)# -pi/2 pi/2 support
    att_responses = np.zeros((np.shape(full_rotations)[0], n, num_cases))
    if num_cases < num_dimensions*num_heights*num_positions:
        num_cases = num_dimensions*num_heights*num_positions
    counter = 0
    for he in range(num_dimensions*num_heights):
        for pos in range(num_positions):
            for gen_idx in range(n):
                label_onehot = np.zeros((1, num_labels))
                label_onehot[:, pos] = 1.0
                label_onehot[:, num_labels - num_dimensions*num_heights + he] = 1.0
                z = tf.random.normal(shape=(1, model.encoder_block1.output[0].shape[1]), mean=0.0, stddev=1.0)
                z_lbl_concat = np.concatenate((z, label_onehot), axis=1)
                preds = model.decoder_block(z_lbl_concat)

                response = tf.reshape(preds[0], [-1, frequency_points])
                response = (response * dataset_std) + dataset_mean
                response = response * dataset_max

                g = np.squeeze(response.numpy())
                g2 = g[:-1]
                # g_inv = np.concatenate((g,g2[::-1]),axis=0)
                att_responses[:, gen_idx, counter] = np.concatenate((g, g2[::-1]), axis=0)
            counter += 1

    dict_1 = {"generated_attenuations_global": att_responses, "rotations": full_rotations,
              "epoch": epoch}

    sio.savemat(
        "results/generated_samples_cvae_ncases_{}_latent_vars_{}_beta_{}_epochs_{}_numheights_{}_numdim_{}.mat".format(num_cases, latent, beta, epoch, num_heights, num_dimensions),
        dict_1)
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(image, cmap='Greys_r')
    # plt.axis('Off')
    # plt.savefig('results/generated_conditoned_digits.png')

    return
def generate_conditioned_attenuations(model,dataset_max,dataset_mean, dataset_std, frequency_points, rotations, epoch, num_classes, beta, height, tm):
    n = 250  # number of generation per input
    latent = model.encoder_block1.output[0].shape[1]
    rotations2 = rotations[:-1]
    full_rotations = np.concatenate((rotations, rotations2[::-1]), axis=0)# -pi/2 pi/2 support
    att_responses = np.zeros((np.shape(full_rotations)[0], n, num_classes))
    for pos in range(num_classes):
        for gen_idx in range(n):
            label_onehot = np.zeros((1,num_classes))
            label_onehot[:, pos] = 1.0

            z = tf.random.normal(shape=(1,model.encoder_block1.output[0].shape[1]),mean=0.0,stddev=1.0)

            z_lbl_concat = np.concatenate((z,label_onehot),axis=1)
            # start = time.time()
            preds = model.decoder_block(z_lbl_concat)
            # end = time.time()
            # print(end-start)
            response = tf.reshape(preds[0],[-1, frequency_points])
            response = (response * dataset_std) + dataset_mean
            response = response * dataset_max

            g = np.squeeze(response.numpy())
            g2 = g[:-1]
            # g_inv = np.concatenate((g,g2[::-1]),axis=0)
            att_responses[:, gen_idx, pos] = np.concatenate((g,g2[::-1]),axis=0)

    dict_1 = {"generated_attenuations": att_responses, "rotations": full_rotations,
              "epoch": epoch}

    # height = 170
    # tm = 45
    sio.savemat(
        "results/generated_samples_cvae_nclasses_{}_latent_vars_{}_beta_{}_epochs_{}_height_{}_trasversal_{}.mat".format(num_classes, latent, beta, epoch, height, tm),
        dict_1)
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(image, cmap='Greys_r')
    # plt.axis('Off')
    # plt.savefig('results/generated_conditoned_digits.png')

    return