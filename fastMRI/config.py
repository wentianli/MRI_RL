import time

class config:
    sampling_scheme = ([0.128], [2.5]) # retain 12.8% low freq, speed-up 2.5 (40% sampling ratio)
    resolution = 320 # 320x320 image
    #-------------learning_related--------------------#
    batch_size = 6
    workers = 4
    iter_size = 4
    num_episodes = 40000
    test_episodes = 1000
    save_episodes = 40000
    resume_model = '' #'model/8_28_21_16000.pth'
    display = 100
    #-------------rl_related--------------------#
    pi_loss_coeff = 1.0
    v_loss_coeff = 0.25
    beta = 0.1
    c_loss_coeff = 0.5 # 0.005
    switch = 8
    warm_up_episodes = 2000
    episode_len = 3
    gamma = 1
    reward_method = 'abs'
    noise_scale = 0.5
    #-------------continuous parameters--------------------#
    actions = {
        'box': 1,
        'bilateral': 2,
        'median': 3,
        'Gaussian': 4,
        'Laplace': 5,
        'Sobel_v1': 6,
        'Sobel_v2': 7,
        'Sobel_h1': 8,
        'Sobel_h2': 9,
        'unsharp': 10,
        'subtraction': 11,
    }
    num_actions = len(actions) + 1

    parameters_scale = {
        'Laplace': 0.2,
        'Sobel_v1': 0.2,
        'Sobel_v2': 0.2,
        'Sobel_h1': 0.2,
        'Sobel_h2':  0.2,
        'unsharp': 1.0,
    }

    #-------------lr_policy--------------------#
    base_lr = 0.001
    # poly
    lr_policy = 'poly'
    policy_parameter = {
      'power': 1,
      'max_iter' : 80000,
    }

    #-------------folder--------------------#
    dataset = 'fastMRI'
    root = '/home/lwt/'
