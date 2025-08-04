# Generative Return Decomposition (GRD) for CartPole-v1
# Based on the paper: "Interpretable Reward Redistribution in Reinforcement Learning: A Causal Approach"
# (Zhang et al., NeurIPS 2023)
# Implemented by Gemini

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel
import numpy as np
import random
from collections import deque, namedtuple
import time
import math

# --- Hyperparameters ---
# Note: These are example hyperparameters and may require tuning for optimal performance.
# The paper uses different values for more complex MuJoCo environments.
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

# --- Environment Wrapper for Episodic Rewards ---
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

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
Trajectory = namedtuple('Trajectory', ('states', 'actions', 'total_return'))

class ReplayBuffer:
    """A replay buffer for storing transitions and trajectories."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.trajectory_memory = deque([], maxlen=capacity // 10) # Store fewer full trajectories

    def push(self, *args):
        self.memory.append(Transition(*args))

    def push_trajectory(self, states, actions, total_return):
        self.trajectory_memory.append(Trajectory(states, actions, total_return))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def sample_trajectories(self, batch_size):
        return random.sample(self.trajectory_memory, min(len(self.trajectory_memory), batch_size))

    def __len__(self):
        return len(self.memory)

# --- Generative Return Decomposition (GRD) Model ---
class CausalModule(nn.Module):
    """
    Learns the causal structure (binary masks C) of the environment.
    This module holds the learnable parameters (logits) for the Bernoulli
    distributions representing the existence of causal links.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Logits for causal masks. Shape: [..., 2] for (Not Exists, Exists)
        self.s_to_s_logits = nn.Parameter(torch.randn(state_dim, state_dim, 2))
        self.a_to_s_logits = nn.Parameter(torch.randn(action_dim, state_dim, 2))
        self.s_to_r_logits = nn.Parameter(torch.randn(state_dim, 2))
        self.a_to_r_logits = nn.Parameter(torch.randn(action_dim, 2))

    def _sample_mask(self, logits, temperature=1.0, hard=True):
        """Samples a binary mask using the Gumbel-Softmax trick."""
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)[:, 1]

    def _sample_mask_2d(self, logits, temperature=1.0, hard=True):
        """Samples a 2D binary mask using Gumbel-Softmax."""
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)[:, :, 1]

    def get_causal_masks(self, training=True, temperature=1.0):
        """
        Get all causal masks.
        During training, use Gumbel-Softmax for differentiable sampling.
        During inference, use argmax (hard=True, no noise) for deterministic masks.
        """
        hard = not training
        
        C_s_s = self._sample_mask_2d(self.s_to_s_logits, temperature, hard)
        C_a_s = self._sample_mask_2d(self.a_to_s_logits, temperature, hard)
        C_s_r = self._sample_mask(self.s_to_r_logits, temperature, hard)
        C_a_r = self._sample_mask(self.a_to_r_logits, temperature, hard)

        return C_s_s, C_a_s, C_s_r, C_a_r

    def get_log_probs(self):
        """Calculate the log probabilities for the regularization loss."""
        # Encourages sparsity by maximizing the log-prob of the "Not Exists" class (index 0)
        log_prob_s_s = F.log_softmax(self.s_to_s_logits, dim=-1)[:, :, 0].sum()
        log_prob_a_s = F.log_softmax(self.a_to_s_logits, dim=-1)[:, :, 0].sum()
        log_prob_s_r = F.log_softmax(self.s_to_r_logits, dim=-1)[:, 0].sum()
        log_prob_a_r = F.log_softmax(self.a_to_r_logits, dim=-1)[:, 0].sum()
        return log_prob_s_s, log_prob_a_s, log_prob_s_r, log_prob_a_r

    def get_compact_representation_mask(self, C_s_s, C_s_r):
        """
        Calculates the compact representation mask (C_s_pi) as per Eq. 7 in the paper.
        A state is important if it directly affects the reward, or if it affects another
        state that is important. This is a transitive closure over the state graph.
        """
        with torch.no_grad():
            s_pi_mask = C_s_r.clone()  # s_pi_mask is float (0.0 or 1.0)
            for _ in range(self.state_dim):  # Iterate to propagate influence
                old_s_pi_mask = s_pi_mask.clone()

                # Ensure matmul operands are float. Result (new_influencers_mask) is bool.
                new_influencers_mask = (C_s_s @ s_pi_mask.float().unsqueeze(1)).squeeze(1) > 0

                # Combine masks using logical OR. Both operands should be bool.
                new_mask = torch.logical_or(s_pi_mask.bool(), new_influencers_mask)
                
                # Convert back to float for the next iteration and for comparison.
                s_pi_mask = new_mask.float()

                # Check for convergence
                if torch.all(s_pi_mask == old_s_pi_mask):
                    break  # Converged
        return s_pi_mask.detach()


class RewardModel(nn.Module):
    """Predicts the unobserved Markovian reward r_t."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, state, action, C_s_r, C_a_r):
        # Apply causal masks
        masked_state = state * C_s_r
        masked_action = action * C_a_r
        
        x = torch.cat([masked_state, masked_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DynamicsModel(nn.Module):
    """Predicts the next state s_{t+1}."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, state_dim)

    def forward(self, state, action, C_s_s, C_a_s):
        # This is a simplification. The paper suggests a more complex component-wise model.
        # For CartPole, a joint model is feasible.
        # BUG FIX: The model was not using the causal masks. Now it does.
        masked_state = state * C_s_s.sum(dim=1) # A simplified way to use the mask
        masked_action = action * C_a_s.sum(dim=1)
        x = torch.cat([masked_state, masked_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predicted_next_state = self.fc3(x)
        return predicted_next_state

class GenerativeModel(nn.Module):
    """The complete GRD generative model."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.causal_module = CausalModule(state_dim, action_dim).to(DEVICE)
        self.reward_model = RewardModel(state_dim, action_dim).to(DEVICE)
        self.dynamics_model = DynamicsModel(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=LR_GENERATIVE)

    def calculate_loss(self, trajectories, transitions):
        L_rew, L_dyn, L_reg = torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)
        
        # --- 1. Reward Redistribution Loss (L_rew) ---
        if trajectories:
            C_s_s, C_a_s, C_s_r, C_a_r = self.causal_module.get_causal_masks(training=True)
            rew_losses = []
            for traj in trajectories:
                states, actions, total_return = traj
                
                actions_one_hot = F.one_hot(actions, num_classes=self.causal_module.action_dim).float()
                predicted_rewards = self.reward_model(states, actions_one_hot, C_s_r, C_a_r)
                
                discounts = torch.pow(GAMMA, torch.arange(len(states), device=DEVICE)).unsqueeze(1)
                predicted_return = torch.sum(predicted_rewards * discounts)
                
                rew_losses.append(F.mse_loss(predicted_return, total_return))
            L_rew = torch.stack(rew_losses).mean()
        
        # --- 2. Dynamics Loss (L_dyn) ---
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=DEVICE, dtype=torch.long)
        next_state_batch = torch.cat(batch.next_state)
        
        action_one_hot_batch = F.one_hot(action_batch, num_classes=self.causal_module.action_dim).float()

        C_s_s, C_a_s, _, _ = self.causal_module.get_causal_masks(training=True)
        predicted_next_states = self.dynamics_model(state_batch, action_one_hot_batch, C_s_s, C_a_s)
        L_dyn = F.mse_loss(predicted_next_states, next_state_batch)

        # --- 3. Regularization Loss (L_reg) ---
        log_prob_s_s, log_prob_a_s, log_prob_s_r, log_prob_a_r = self.causal_module.get_log_probs()
        L_reg = -(LAMBDA_S_S * log_prob_s_s + LAMBDA_A_S * log_prob_a_s + \
                  LAMBDA_S_R * log_prob_s_r + LAMBDA_A_R * log_prob_a_r)
        
        total_loss = L_rew + L_dyn + L_reg
        return total_loss, L_rew, L_dyn, L_reg

# --- Discrete Soft Actor-Critic (SAC) Agent ---
class Actor(nn.Module):
    """SAC Actor for discrete action spaces."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits

    def get_action_dist(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)

class Critic(nn.Module):
    """
    SAC Critic. Outputs Q-values for all actions.
    CRITICAL FIX: Changed output from 1 to action_dim.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1 architecture
        self.fc1_q1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2_q1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3_q1 = nn.Linear(HIDDEN_DIM, action_dim)

        # Q2 architecture
        self.fc1_q2 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2_q2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3_q2 = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state):
        # Q1
        x1 = F.relu(self.fc1_q1(state))
        x1 = F.relu(self.fc2_q1(x1))
        q1 = self.fc3_q1(x1)
        # Q2
        x2 = F.relu(self.fc1_q2(state))
        x2 = F.relu(self.fc2_q2(x2))
        q2 = self.fc3_q2(x2)
        return q1, q2

class DiscreteSACAgent:
    def __init__(self, state_dim, action_dim, compact_state_dim):
        self.actor = Actor(compact_state_dim, action_dim).to(DEVICE)
        # CRITICAL FIX: Critic now takes action_dim
        self.critic = Critic(compact_state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(compact_state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_POLICY)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_POLICY)
        
        self.action_dim = action_dim

    def select_action(self, compact_state):
        with torch.no_grad():
            dist = self.actor.get_action_dist(compact_state)
            action = dist.sample()
        return action.item()

    def update(self, transitions, generative_model):
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=DEVICE, dtype=torch.long).unsqueeze(1)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float).unsqueeze(1)

        with torch.no_grad():
            # Get causal masks for compact representation and reward prediction
            C_s_s, _, C_s_r, C_a_r = generative_model.causal_module.get_causal_masks(training=False)
            compact_mask = generative_model.causal_module.get_compact_representation_mask(C_s_s, C_s_r)
            
            # Use the learned reward model to get Markovian rewards
            action_one_hot = F.one_hot(action_batch.squeeze(1), num_classes=self.action_dim).float()
            redistributed_rewards = generative_model.reward_model(state_batch, action_one_hot, C_s_r, C_a_r)
            
            # Apply compact representation mask
            compact_next_state = next_state_batch * compact_mask
            
            # --- Calculate target Q value ---
            next_action_dist = self.actor.get_action_dist(compact_next_state)
            next_action_probs = next_action_dist.probs
            next_log_probs = torch.log(next_action_probs + 1e-8)

            q1_next_target, q2_next_target = self.critic_target(compact_next_state)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # For discrete SAC, the next state value is the expectation over next actions
            next_value = torch.sum(next_action_probs * (min_q_next_target - ALPHA * next_log_probs), dim=1, keepdim=True)
            
            target_q = redistributed_rewards + (1 - done_batch) * GAMMA * next_value

        # --- Critic Update ---
        compact_state = state_batch * compact_mask
        current_q1, current_q2 = self.critic(compact_state)
        
        # CRITICAL FIX: Gather Q-values for the actions that were actually taken
        q1_taken_action = current_q1.gather(1, action_batch)
        q2_taken_action = current_q2.gather(1, action_batch)
        
        critic_loss = F.mse_loss(q1_taken_action, target_q) + F.mse_loss(q2_taken_action, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        action_dist = self.actor.get_action_dist(compact_state)
        action_probs = action_dist.probs
        log_probs = torch.log(action_probs + 1e-8)
        
        # We need Q-values for the actor loss calculation
        q1_val, q2_val = self.critic(compact_state)
        min_q = torch.min(q1_val, q2_val).detach() # Detach to stop gradients to critic
        
        actor_loss = torch.sum(action_probs * (ALPHA * log_probs - min_q), dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()


# --- Main Training Loop ---
def main():
    env = gym.make("CartPole-v1", render_mode="human")
    env = EpisodicRewardWrapper(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    compact_state_dim = state_dim 

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    generative_model = GenerativeModel(state_dim, action_dim)
    sac_agent = DiscreteSACAgent(state_dim, action_dim, compact_state_dim)

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

if __name__ == '__main__':
    main()
