import argparse
import gym
import torch
import numpy as np

from sac.sac import SAC

def readParser():
    parser = argparse.ArgumentParser(description='DAGBNN')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--actor_path', default="/",
                        help='Path to the actor')
    parser.add_argument('--critic_path', default="/",
                        help='Path to the critic')                  
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--random', action='store_true')

    return parser.parse_args()

def main(args=None):
    
    if args is None:
        args = readParser()

    # Initial environment
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Initial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_model(args.actor_path, args.critic_path)

    vis_env = gym.make(args.env_name)
    while True:
        done = False
        obs = vis_env.reset()
        vis_env.render()
        while not done:
            if args.random:
                action = env.action_space.sample() # take a random action
            else:
                action = agent.select_action(obs)
            obs, reward, done, info = vis_env.step(action)
            vis_env.render()
        print(f"Reward: {reward}")

if __name__ == '__main__':
    main()