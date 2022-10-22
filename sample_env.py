import gym
import numpy as np

class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.action_1 = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0
        self.action_length = np.prod(env.action_space.shape)

    def sample(self, agent, eval_t=False):
        
        if self.current_state is None:
            self.current_state = self.env.reset()
            self.action_1 = np.zeros(self.action_length)
            
        cur_state = self.current_state
        action_1 = self.action_1
        action = agent.select_action(cur_state, eval_t)
        next_state, reward, terminal, info = self.env.step(action)

        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
            # self.action_1 = None
        else:
            self.current_state = next_state
            self.action_1 = action

        return cur_state, action, action_1, next_state, reward, terminal, info
