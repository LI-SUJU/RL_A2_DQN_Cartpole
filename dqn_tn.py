import gym as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from plotHelper import smooth
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

LAYER_COUNT = 1 # need to tune, 3, 4, 5
HIDDEN_DIM = 128 # need to tune, 16, 32, 64

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # self.layer1 = nn.Linear(n_observations, 128)
        # self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, n_actions)
                # Define layers with ReLU activation
        self.linear1 = nn.Linear(n_observations, HIDDEN_DIM)
        # self.activation1 = nn.ReLU()
        self.activation1 = nn.ReLU()
        for i in range(LAYER_COUNT):
            setattr(self, f'linear{i+2}', nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            setattr(self, f'activation{i+2}', nn.ReLU())
        # self.linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # # self.activation2 = nn.ReLU()
        # self.activation2 = nn.LeakyReLU()
        # self.linear3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # # self.activation3 = nn.ReLU()
        # self.activation3 = nn.LeakyReLU()

        # Output layer without activation function
        self.output_layer = nn.Linear(HIDDEN_DIM, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.activation1(self.linear1(x))
        for i in range(LAYER_COUNT):
            x = getattr(self, f'activation{i+2}')(getattr(self, f'linear{i+2}')(x))
        x = self.output_layer(x)
        return x
    
    def get_layer_count(self):
        return LAYER_COUNT
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 1000
def training(num_episodes):
    
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′

            if done:
                episode_durations.append(t + 1)
                if i_episode % 10 == 0:
                    print(f'Episode {i_episode}, reward: {t+1}')
                # plot_durations()
                break
    return episode_durations      

# Plotting
plt.figure(figsize=(10, 6))
# set the font size
plt.rcParams.update({'font.size': 15})

episode_durations = training(num_episodes)
# if the folder does not exist, create it
import os
os.makedirs('./data4plot', exist_ok=True)
# Save episode_durations as a file
np.savetxt('./data4plot/dqn_tn_episode_durations.txt', episode_durations)
plt.plot(episode_durations, alpha=0.1, color="orange")
# please plot a line using smooth() function from plotHelper.py
plt.plot(smooth(episode_durations, 30), label=f'Layers Number: {LAYER_COUNT}', alpha=1.0, color="orange")
########################################################################################
# plt.plot(range(49, len(episode_durations), 50), [sum(episode_durations[i:i+50])/50 for i in range(0, len(episode_durations), 50)], alpha=1.0, color="orange")
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Comparisons of Networks with Different Number of Layers')
plt.legend()
# Add text
text = f'Learning Rate: {LR}, Exploration Policy: Epsilon-Greedy,\nDimention of Hidden Layers: {HIDDEN_DIM}, Gamma: {GAMMA}, Tau: {TAU}'
plt.text(0.02, 80, text, verticalalignment='top', fontsize=12, alpha=0.5)
# if the folder does not exist, create it
os.makedirs('./plots/dqn-tn', exist_ok=True)
# Save plot
plt.savefig(f'./plots/dqn-tn/DQN_{num_episodes}.png')

# test the model
# env = gym.make("CartPole-v1")
# state, info = env.reset()
# state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
# # print the reward
# reward = 0
# mean_reward = []
# # test for 20 epochs, give the mean reward
# for i in range(20):
#     for t in count():
#         action = policy_net(state).max(1).indices.view(1, 1)
#         observation, r, terminated, truncated, _ = env.step(action.item())
#         reward += r
#         if terminated or truncated:
#             state, info = env.reset()
#             state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#             break
#         else:
#             state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
#     mean_reward.append(reward) 
#     reward = 0
# # print the mean reward
# # get the mean of mean_reward[]
# # plot the mean_reward
    

#     plt.figure(2)
#     plt.title('Mean Reward')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Reward')
#     plt.plot(mean_reward)
#     plt.show()

# mean_reward = sum(mean_reward)/len(mean_reward)

# print(mean_reward)

print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()