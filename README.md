# generative_diffraction_model
 the code can be used to generate RF attenuation samples according to a diffraction model using a pre-trained C-VAE network. 
 
Pre-trained models limitations: transmitter-receiver distance (4m) and height from ground 1m, 2.40GHz, maximum and minimum target height (2m and 1.7m), maximum and minimum target trasversal max size (0.55m) and 0.45m. 75 nominal positions used in training (see nominal_positions.mat)

Pre-trained models are ordered by latent dimensions and beta

Usage:

real_time_generation_cvae_soft_labels.py [-h]   [-latent_dim LATENT_DIM]
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
  
  -random_height_dim RANDOM_HEIGHT_DIM set to 1 for random height and dimension generation, 0 to set an assigned target size and variable positions in nominal_position.mat, 2 for assigned target size, and relative position (x,y) to the transmitter (distance 4m)
  
  -height HEIGHT        set the target height in m
  
  -tm TM                set the target trasversal max size in m
  
  -H1 H1                set the MAX target height in m
  
  -T1 T1                set the MAX target trasversal max size in m
  
  -H2 H2                set the MIN target height in m
  
  -T2 T2                set the MIN target trasversal max size in m
  
  -pos_x POS_X          set the target position in x domain (along the los) in m
  
  -pos_y POS_Y          set the target position in y domain (across the los) in m
