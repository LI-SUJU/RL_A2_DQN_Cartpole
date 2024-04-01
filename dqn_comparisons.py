import os
from plotHelper import smooth

import matplotlib.pyplot as plt

LAYER_COUNT = 1 # need to tune, 3, 4, 5
HIDDEN_DIM = 128 # need to tune, 16, 32, 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get the list of files in the directory
directory = './data4plot'
files = os.listdir(directory)
# Plotting
plt.figure(figsize=(10, 6))
# set the font size
plt.rcParams.update({'font.size': 15})

# Iterate over the files and plot the rewards
for file in files:
    # Read the rewards from the file
    rewards = []
    with open(os.path.join(directory, file), 'r') as f:
        for line in f:
            rewards.append(float(line.strip()))

    # Smooth the rewards using the smooth() function from plotHelper
    smoothed_rewards = smooth(rewards, 30)

    # Plot the original rewards with transparency alpha of 0.5
    # ax.plot(rewards, alpha=0.1)

    # Plot the smoothed rewards with a solid line, using the file name as the label, use orange for DQN, blue for DQN with ER, and green for DQN with TN, and red for DQN with ER and TN, respectively, make legend
    if 'dqn_er_tn' in file:
        plt.plot(smoothed_rewards, label='DQN-ER-TN', color='firebrick')
    elif 'dqn_er' in file:
        plt.plot(smoothed_rewards, label='DQN-ER', color='navy')
    elif 'dqn_tn' in file:
        plt.plot(smoothed_rewards, label='DQN-TN', color='olive')
    else:
        continue
        # plt.plot(smoothed_rewards, label='DQN', color='orange')

# Add text
text = f'Learning Rate: {LR}, Exploration Policy: Epsilon-Greedy, Number of Layers: {LAYER_COUNT},\nDimention of Hidden Layers: {HIDDEN_DIM}, Gamma: {GAMMA}, Tau: {TAU}'
plt.text(0.02, 20, text, verticalalignment='top', fontsize=12, alpha=0.5)

# Add legends, labels, and a title to the plot
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Comparisons of DQN Variants')

# Save the plot to a file
# plt.savefig('./plots/dqn_comparison/dqn_comparisons.png')

plt.savefig('./plots/dqn_comparison/dqn_comparisons_without_dqn.png')