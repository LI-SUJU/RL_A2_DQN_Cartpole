# Description: This file contains the epsilon decay function.
# Usage: please import this file in the main file and use the function decay_epsilon(episode, current_reward, eps, MODE, decay_rate) to decay the epsilon.
# the parameters you need to pass are: (feel free to change the values)
# this file is mainly referred to https://github.com/aakash94/RED/blob/master/Balancer.ipynb

EPSILON    = 1
EPSILON_DECAY    = 0.999
MODE_E = 'EXPONENTIAL'
MODE_L = 'LINEAR'
MODE_R = 'RBED'
def linear_decay( episode, decay_till=1024 , start_val = 1.0, end_val = 0.01):
    e = (((end_val - start_val)/decay_till)*episode) + start_val
    return max(e, end_val)

REWARD_THRESHOLD = 0
def reward_based_e_decay(current_reward, eps, highest_ep, lowest_ep, target_reward=500, target_increment=1):
        
    global REWARD_THRESHOLD
    
    
#     if eps < lowest_ep:
#         return eps
    
#     if eps < stop_red :
#         eps *= decay_rate
#         return rps
    
    steps_to_move_in = target_reward
    #steps_to_move_in = steps_to_move
    quanta = (highest_ep - lowest_ep)/steps_to_move_in   
    
    
    #REWARD_THRESHOLD = min(REWARD_THRESHOLD, 200)
    
    if current_reward > REWARD_THRESHOLD :
        REWARD_THRESHOLD += target_increment
        eps -= quanta
        
    return max(eps, lowest_ep)

REWARD_THRESHOLD = 0
def reward_based_e_decay1(current_reward, eps=EPSILON, highest_ep=1.0, lowest_ep=0.01, target_reward=500, target_increment=1, steps_to_move=256):
    
    steps_to_move_in = target_increment
    # steps_to_move_in = steps_to_move
    
    quanta = (highest_ep - lowest_ep)/steps_to_move_in
        
    return max(eps, lowest_ep)

def exponential_decay (eps=EPSILON, decay_rate=EPSILON_DECAY):
    # need to decide decay_rate.
    eps *= decay_rate
    return  eps

# export the function
def decay_epsilon(episode, current_reward=0, eps=EPSILON, MODE=MODE_L, decay_rate=EPSILON_DECAY, decay_till=1024, start_val=1.0, end_val=0.01, target_reward=500, target_increment=1):
    
    if MODE == MODE_L:
        eps = linear_decay(episode, decay_till , start_val, end_val)
    
    elif MODE == MODE_E:
        eps = exponential_decay(eps, decay_rate)
    
    elif MODE == MODE_R:
        eps = reward_based_e_decay(current_reward = current_reward, eps=EPSILON, highest_ep=start_val, lowest_ep=end_val, target_reward=target_reward, target_increment=target_increment)
    
    return eps



