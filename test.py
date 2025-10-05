import gymnasium as gym
import torch
import time
import numpy as np
import torch.nn.functional as F
from modules import RewardModel, GenerativeModel, ReplayBuffer, DiscreteSACAgent, EpisodicRewardWrapper


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_POLICY = 3e-4          # Learning rate for actor and critic
LR_GENERATIVE = 3e-4      # Learning rate for generative model
GAMMA = 0.99              # Discount factor
REPLAY_BUFFER_SIZE = 50000 # Size of the replay buffer
BATCH_SIZE = 256          # Batch size for training
TAU = 0.005               # Soft update coefficient for target networks
ALPHA = 0.2               # SAC temperature parameter (entropy regularization)
HIDDEN_DIM = 256          # Hidden dimension for neural networks
MAX_EPISODES = 1000       # Total number of episodes to run
MAX_STEPS_PER_EPISODE = 500 # Max steps per episode for CartPole-v1
START_TRAINING_EPISODES = 10 # Number of episodes to collect data before training starts

# Hyperparameters for the GRD generative model loss (L_reg)
# These control the sparsity of the learned causal graph. Increased to encourage sparsity.
LAMBDA_S_R = 5e-4  # state -> reward
LAMBDA_A_R = 1e-5  # action -> reward
LAMBDA_S_S = 5e-5  # state -> state
LAMBDA_A_S = 1e-8  # action -> state

def main():
    env = gym.make("CartPole-v1")
    env = EpisodicRewardWrapper(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    compact_state_dim = state_dim 

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    generative_model = GenerativeModel(state_dim, action_dim, HIDDEN_DIM, DEVICE, LR_GENERATIVE, GAMMA, LAMBDA_S_S, LAMBDA_S_R, LAMBDA_A_S, LAMBDA_A_R)
    sac_agent = DiscreteSACAgent(state_dim, action_dim, compact_state_dim, HIDDEN_DIM, DEVICE, LR_POLICY, TAU, GAMMA, ALPHA)

    print(f"Using device: {DEVICE}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    start_time = time.time()
    all_episode_steps = []

    for i_episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        episode_reward = 0
        episode_states = []
        episode_actions = []
        
        for t in range(MAX_STEPS_PER_EPISODE):
            with torch.no_grad():
                C_s_s, _, C_s_r, _ = generative_model.causal_module.get_causal_masks(training=False)
                compact_mask = generative_model.causal_module.get_compact_representation_mask(C_s_s, C_s_r)
                compact_state = state * compact_mask

            action = sac_agent.select_action(compact_state)
            
            next_state_np, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # The episodic reward is only non-zero at the end
            # The immediate reward for the replay buffer is this episodic reward if done, else 0
            replay_reward = reward if done else 0.0
            episode_reward += reward 
            
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            # Push the transition with the immediate (mostly zero) reward
            replay_buffer.push(state, action, next_state, replay_reward, done)
            episode_states.append(state)
            episode_actions.append(action)
            
            state = next_state

            if len(replay_buffer) > BATCH_SIZE and i_episode > START_TRAINING_EPISODES:
                # Update Generative Model
                generative_model.optimizer.zero_grad()
                transitions = replay_buffer.sample(BATCH_SIZE)
                trajectories = replay_buffer.sample_trajectories(4)
                gen_loss, l_rew, l_dyn, l_reg = generative_model.calculate_loss(trajectories, transitions)
                if torch.is_tensor(gen_loss):
                    gen_loss.backward()
                    generative_model.optimizer.step()

                # Update Policy Model
                actor_loss, critic_loss = sac_agent.update(transitions, generative_model)

            if done:
                break
        
        all_episode_steps.append(t + 1)

        # After episode ends, store the full trajectory with the final episodic return
        if episode_reward > 0:
            ep_states_tensor = torch.cat(episode_states, dim=0)
            ep_actions_tensor = torch.tensor(episode_actions, device=DEVICE, dtype=torch.long)
            ep_return_tensor = torch.tensor([episode_reward], dtype=torch.float32, device=DEVICE)
            replay_buffer.push_trajectory(ep_states_tensor, ep_actions_tensor, ep_return_tensor)

        if i_episode % 1 == 0:
            avg_steps = np.mean(all_episode_steps[-100:]) # Average over last 100 episodes
            elapsed_time = time.time() - start_time
            print(f"Epi {i_episode}/{MAX_EPISODES} | Avg Steps (last 100): {avg_steps:.2f} | Steps: {t+1} | Time: {elapsed_time:.2f}s")
            print(f"Reward: {episode_reward}")
            if len(replay_buffer) > BATCH_SIZE and i_episode > START_TRAINING_EPISODES and torch.is_tensor(gen_loss):
                print(f"  Losses -> Gen: {gen_loss.item():.4f} (Rew: {l_rew.item():.4f}, Dyn: {l_dyn.item():.4f}, Reg: {l_reg.item():.4f})")
                print(f"  Policy -> Actor: {actor_loss:.4f}, Critic: {critic_loss:.4f}")

            with torch.no_grad():
                C_s_s, _, C_s_r, C_a_r = generative_model.causal_module.get_causal_masks(training=False)
                compact_mask = generative_model.causal_module.get_compact_representation_mask(C_s_s, C_s_r)
                s_r_probs = F.softmax(generative_model.causal_module.s_to_r_logits, dim=-1)[:, 1].cpu().numpy()
                a_r_probs = F.softmax(generative_model.causal_module.a_to_r_logits, dim=-1)[:, 1].cpu().numpy()
                print(f"  Causal Probs (S->R): {[f'{p:.2f}' for p in s_r_probs]}")
                print(f"  Compact Mask: {compact_mask.cpu().numpy()}")

    env.close()
    
    sac_agent.save('weights')


if __name__ == "__main__":
    main()