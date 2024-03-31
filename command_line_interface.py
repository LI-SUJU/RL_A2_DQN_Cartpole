import argparse
import runpy
def main():
    parser = argparse.ArgumentParser(description='DQN Command Line Interface')
    parser.add_argument('--experience_replay', action='store_true', help='Enable experience replay')
    parser.add_argument('--target_network', action='store_true', help='Enable target network')
    parser.add_argument('--echo', action='store_true', help='Print help message')
    args = parser.parse_args()
    print(args)
    if args.experience_replay and args.target_network:
        run_dqn('dqn_er_tn.py')
    elif args.experience_replay:
        run_dqn('dqn_er.py')
    elif args.target_network:
        run_dqn('dqn_tn.py')
    elif args.help:
        parser.print_help()
    else:
        run_dqn('dqn.py')

def run_dqn(file_name):
    # Run the specified DQN file
    print(f"Running {file_name}...")
    # Add your code here to execute the specified DQN file
    runpy.run_path(file_name)
    print(f"{file_name} executed successfully! You can check the results in the ./plots folder.")

if __name__ == '__main__':
    main()