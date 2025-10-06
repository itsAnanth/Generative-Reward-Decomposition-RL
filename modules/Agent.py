import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from .Transition import Transition

# --- Discrete Soft Actor-Critic (SAC) Agent ---
class Actor(nn.Module):
    """SAC Actor for discrete action spaces."""
    def __init__(self, state_dim, action_dim, HIDDEN_DIM):
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
    def __init__(self, state_dim, action_dim, HIDDEN_DIM):
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
    def __init__(self, state_dim, action_dim, compact_state_dim, 
                HIDDEN_DIM,
                DEVICE, 
                LR_POLICY,
                TAU,
                GAMMA,
                ALPHA
            ):
        self.actor = Actor(compact_state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        # CRITICAL FIX: Critic now takes action_dim
        self.critic = Critic(compact_state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.critic_target = Critic(compact_state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_POLICY)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_POLICY)
        
        self.action_dim = action_dim
        self.DEVICE = DEVICE
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA

    def select_action(self, compact_state):
        with torch.no_grad():
            dist = self.actor.get_action_dist(compact_state)
            action = dist.sample()
        return action.item()

    def update(self, transitions, generative_model):
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=self.DEVICE, dtype=torch.long).unsqueeze(1)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor(batch.done, device=self.DEVICE, dtype=torch.float).unsqueeze(1)

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
            next_value = torch.sum(next_action_probs * (min_q_next_target - self.ALPHA * next_log_probs), dim=1, keepdim=True)
            
            target_q = redistributed_rewards + (1 - done_batch) * self.GAMMA * next_value

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
        
        actor_loss = torch.sum(action_probs * (self.ALPHA * log_probs - min_q), dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()

    def save(self, save_dir):
        """Save the actor and critic model parameters."""
        torch.save(self.actor.state_dict(), f"{save_dir}/actor.pth")
        torch.save(self.critic.state_dict(), f"{save_dir}/critic.pth")
        torch.save(self.critic_target.state_dict(), f"{save_dir}/critic_target.pth")
        
    def load(self, save_dir, device):
        self.actor.load_state_dict(torch.load(f'{save_dir}/actor.pth', map_location=device, weights_only=True))
        self.critic.load_state_dict(torch.load(f'{save_dir}/critic.pth', map_location=device, weights_only=True))
        self.critic_target.load_state_dict(torch.load(f'{save_dir}/critic_target.pth', map_location=device, weights_only=True))
        