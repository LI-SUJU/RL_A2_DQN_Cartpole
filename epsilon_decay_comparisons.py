import os

import matplotlib.pyplot as plt

# Create the plots directory if it doesn't exist
plots_dir = './plots/epsilon_decay'
os.makedirs(plots_dir, exist_ok=True)

# Read files in the ./data4plot/epsilon directory
epsilon_dir = './data4plot/epsilon'
files = os.listdir(epsilon_dir)

# Initialize lists to store data for plotting
epsilon_data = []
reward_data = []

# Process each file
for file in files:
    file_path = os.path.join(epsilon_dir, file)
    
    # Check if the file name contains '_eps'
    if '_eps' in file:
        # Read epsilon data from the file
        with open(file_path, 'r') as f:
            epsilon_values = [float(line.strip()) for line in f]
        epsilon_data.append(epsilon_values)
    else:
        # Read reward data from the file
        with open(file_path, 'r') as f:
            reward_values = [float(line.strip()) for line in f]
        reward_data.append(reward_values)

# Plot epsilon data
if epsilon_data:
    plt.figure()
    for epsilon_values in epsilon_data:
        plt.plot(epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.savefig(os.path.join(plots_dir, 'epsilon_decay.png'))
    plt.close()

# Plot reward data
if reward_data:
    plt.figure()
    for reward_values in reward_data:
        plt.plot(reward_values)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(plots_dir, 'epsilon_reward.png'))
    plt.close()