configurations = {
    # ENVIRONMENT PARAMETERS
    'reward_penalty': [0.001, 0.005, 0.1],
    'naction_ending': [5, 4, 3, 6],
    'step_sampling': [True, False],
    'offline_max_remove_data_perc': [0.4, 0.3, 0.1],
    'offline_max_remove_network': [1, 2, 0],

    # DQN TRAINING PARAMS
    'BATCH_SIZE': [50, 100], 
    'GAMMA': [0.99, 0.5], 
    'EPS_START': [0.9], 
    'EPS_END': [0.05, 0.1], 
    'EPS_DECAY': [200], 
    'TARGET_UPDATE': [5],
    

}




# reward_penalty=0.001,
# naction_ending=5, 
# step_sampling=SHUFFLE_SAMPLES, 
# offline_max_remove_data_perc = 0.4,
# offline_max_remove_network = 1

# BATCH_SIZE = 50
# GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 5