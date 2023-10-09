# generative_diffraction_model


real_time_generation_cvae.py: the code can be used to generate RF attenuation samples according to a diffraction model using a pre-trained C-VAE network. 

see paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9538985

Pre-trained models limitations: transmitter-receiver distance (4m) and height from ground 1m, 2.40GHz, maximum and minimum target height (2m and 1.7m), maximum and minimum target trasversal max size (0.55m) and 0.45m. 75 nominal positions used in training (see nominal_positions.mat)

Pre-trained models are ordered by latent dimensions and beta

Usage:

real_time_generation_cvae.py [-h]   [-latent_dim LATENT_DIM]
                                                [-beta BETA]
                                                [-random_height_dim RANDOM_HEIGHT_DIM]
                                                [-height HEIGHT] [-tm TM]
                                                [-H1 H1] [-T1 T1] [-H2 H2]
                                                [-T2 T2] [-pos_x POS_X]
                                                [-pos_y POS_Y]

optional arguments:
  
  -h, --help            show this help message and exit
  
  -latent_dim LATENT_DIM  set the number of latent dimensions for C-VAE network generation and reconstruction
  
  -beta BETA            set the beta weighting for KL divergence
  
  -random_height_dim RANDOM_HEIGHT_DIM set to 0 to generate attenuations for random target height (H2<height<H1), dimension (T2<tm<T1) and variable positions in nominal_positions.mat, 1 to set an assigned target size (height, tm) and variable positions in nominal_position.mat, 2 for assigned target size (height, tm), and relative position (x,y) to the transmitter (distance 4m)
  
  -height HEIGHT        set the target height in m
  
  -tm TM                set the target trasversal max size in m
  
  -H1 H1                set the MAX target height in m
  
  -T1 T1                set the MAX target trasversal max size in m
  
  -H2 H2                set the MIN target height in m
  
  -T2 T2                set the MIN target trasversal max size in m
  
  -pos_x POS_X          set the target position in x domain (along the los) in m
  
  -pos_y POS_Y          set the target position in y domain (across the los) in m

real_time_generation_cvae_MIMO.py: the code can be used to generate RF attenuation samples now considering a MIMO (multiple input multiple output transmitter-receiver setup) 

Pre-trained models limitations: 9 antenna at the TX, 9 antennas at the RX (spacing half-wavelength); transmitter-receiver distance (4m) and height from ground 1m, 2.40GHz, maximum and minimum target height (2m and 1.7m), maximum and minimum target trasversal max size (0.55m) and 0.45m. 75 nominal positions used in training (see nominal_positions.mat)

Pre-trained models are ordered by latent dimensions and beta

Before using unzip the file cvae.ckpt (which contains the pre-trained model parameters) in the corresponding folder

usage: real_time_generation_cvae_MIMO.py [-h] [-latent_dim LATENT_DIM]
                                         [-beta BETA]
                                         [-random_height_dim RANDOM_HEIGHT_DIM]
                                         [-gen GEN] [-height HEIGHT] [-tm TM]
                                         [-H1 H1] [-T1 T1] [-H2 H2] [-T2 T2]
                                         [-pos_x POS_X] [-pos_y POS_Y]

optional arguments:
  -h, --help            show this help message and exit
  -latent_dim LATENT_DIM
                        set the number of latent dimensions for reconstruction
  -beta BETA            set the beta weighting for KL divergence
  -random_height_dim RANDOM_HEIGHT_DIM
                        1 to set an assigned target size and variable
                        positions in nominal_position.mat, 2 for assigned
                        target size, and position
  -gen GEN              set the number of generated samples
  -height HEIGHT        set the target height
  -tm TM                set the target trasversal max size
  -H1 H1                set the MAX target height
  -T1 T1                set the MAX target trasversal max size
  -H2 H2                set the MIN target height
  -T2 T2                set the MIN target trasversal max size
  -pos_x POS_X          set the target position in x domain (along the los)
  -pos_y POS_Y          set the target position in y domain (across the los)