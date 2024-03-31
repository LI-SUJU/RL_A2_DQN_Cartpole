from matplotlib import pyplot as plt
from dqn import training as dqn_training
from dqn_er import training as dqn_er_training
from dqn_tn import training as dqn_tn_training
from dqn_er_tn import training as dqn_er_tn_training
from plotHelper import smooth

num_episodes = 700
LAYER_COUNT = 1
HIDDEN_DIM = 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
# Plotting
plt.figure(figsize=(10, 6))
# set the font size
plt.rcParams.update({'font.size': 15})

episode_durations = dqn_training(num_episodes)
plt.plot(episode_durations, alpha=0.1, color="orange")
# please plot a line using smooth() function from plotHelper.py
plt.plot(smooth(episode_durations, 30), label=f'DQN', alpha=1.0, color="orange")
########################################################################################
episode_durations = []
episode_durations = dqn_er_training(num_episodes)
plt.plot(episode_durations, alpha=0.1, color="navy")
plt.plot(smooth(episode_durations, 30), label=f'DQN-ER', alpha=1.0, color="navy")
########################################################################################
episode_durations = []
episode_durations = dqn_tn_training(num_episodes)
plt.plot(episode_durations, alpha=0.1, color="olive")
plt.plot(smooth(episode_durations, 30), label=f'DQN-TN', alpha=1.0, color="olive")
########################################################################################
episode_durations = []
episode_durations = dqn_er_tn_training(num_episodes)
plt.plot(episode_durations, alpha=0.1, color="firebrick")
plt.plot(smooth(episode_durations, 30), label=f'DQN-ER-TN', alpha=1.0, color="firebrick")
########################################################################################
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Comparisons of DQN Variants')
plt.legend()
# Add text
text = f'Learning Rate: {LR}, Exploration Policy: Epsilon-Greedy,\nDimention of Hidden Layers: {HIDDEN_DIM}, Gamma: {GAMMA}, Tau: {TAU}'
plt.text(0.02, 80, text, verticalalignment='top', fontsize=12, alpha=0.5)
# Save plot
plt.savefig(f'./plots/dqn_comparison/DQN_comparison.png')