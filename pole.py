# %%
import gymnasium as gym
import numpy as np


# %%
def environment_info(env):
    ''' Prints info about the given environment. '''
    print('************** Environment Info **************')
    print('Observation space: {}'.format(env.observation_space))
    print('Observation space high values: {}'.format(env.observation_space.high))
    print('Observation space low values: {}'.format(env.observation_space.low))
    print('Action space: {}'.format(env.action_space))
    print()


# %%
class AgentBasic(object):
    ''' Simple agent class. '''
    def __init__(self):
        pass

    def act(self, obs):
        if len(obs) == 2:
            obs_array, _ = obs  # unpack observation and ignore info
        else:
            obs_array = obs
        angle = obs_array[2]
        return 0 if angle < 0 else 1


# %%
def basic_guessing_policy(env, agent):
    totals = []
    for episode in range(500):
        
        episode_rewards = 0
        obs = env.reset()
        done = False
        # env.render()
        while not done:  # 1000 steps max unless failure
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_rewards += reward
            # env.render()
            done = terminated
        totals.append(episode_rewards)
        print(f"Episode ---- {episode + 1} | reward ---- {episode_rewards}")
        

    print('************** Reward Statistics **************')
    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))

# %%
def main():
    env = gym.make('CartPole-v1', render_mode='human')

    # environment_info(env)
    
    agent = AgentBasic()
    basic_guessing_policy(env, agent)
    env.close()
    
    

# %%
main()

# %%



