import gymnasium as gym

class EpisodicRewardWrapper(gym.Wrapper):
    """
    A wrapper to convert the standard CartPole reward to an episodic one.
    The agent receives a reward only at the end of the episode, which is the
    total number of steps it survived. This creates the "delayed reward" problem
    that GRD is designed to solve.
    """
    def __init__(self, env):
        super().__init__(env)
        self.total_reward = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        
        done = terminated or truncated
        
        if done:
            reward = self.total_reward
        else:
            reward = 0.0
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.total_reward = 0
        return self.env.reset(**kwargs)
