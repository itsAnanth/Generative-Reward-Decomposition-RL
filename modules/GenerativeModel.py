import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .Causal import CausalModel
from .Reward import RewardModel
from .Dynamics import DynamicsModel
from .Transition import Transition

class GenerativeModel(nn.Module):
    """The complete GRD generative model."""
    def __init__(self, state_dim, action_dim, 
                HIDDEN_DIM,
                DEVICE, 
                LR_GENERATIVE, 
                GAMMA,
                LAMBDA_S_S,
                LAMBDA_S_R,
                LAMBDA_A_S,
                LAMBDA_A_R,
            ):
        super().__init__()
        self.causal_module = CausalModel(state_dim, action_dim).to(DEVICE)
        self.reward_model = RewardModel(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.dynamics_model = DynamicsModel(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=LR_GENERATIVE)
        self.DEVICE = DEVICE
        self.GAMMA = GAMMA
        self.LAMBDA_S_S = LAMBDA_S_S
        self.LAMBDA_S_R = LAMBDA_S_R
        self.LAMBDA_A_S = LAMBDA_A_S
        self.LAMBDA_A_R = LAMBDA_A_R


    def calculate_loss(self, trajectories, transitions):
        L_rew, L_dyn, L_reg = torch.tensor(0.0, device=self.DEVICE), torch.tensor(0.0, device=self.DEVICE), torch.tensor(0.0, device=self.DEVICE)
        
        # --- 1. Reward Redistribution Loss (L_rew) ---
        if trajectories:
            C_s_s, C_a_s, C_s_r, C_a_r = self.causal_module.get_causal_masks(training=True)
            rew_losses = []
            for traj in trajectories:
                states, actions, total_return = traj
                
                actions_one_hot = F.one_hot(actions, num_classes=self.causal_module.action_dim).float()
                predicted_rewards = self.reward_model(states, actions_one_hot, C_s_r, C_a_r)
                
                discounts = torch.pow(self.GAMMA, torch.arange(len(states), device=self.DEVICE)).unsqueeze(1)
                predicted_return = torch.sum(predicted_rewards * discounts)
                
                rew_losses.append(F.mse_loss(predicted_return, total_return))
            L_rew = torch.stack(rew_losses).mean()
        
        # --- 2. Dynamics Loss (L_dyn) ---
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=self.DEVICE, dtype=torch.long)
        next_state_batch = torch.cat(batch.next_state)
        
        action_one_hot_batch = F.one_hot(action_batch, num_classes=self.causal_module.action_dim).float()

        C_s_s, C_a_s, _, _ = self.causal_module.get_causal_masks(training=True)
        predicted_next_states = self.dynamics_model(state_batch, action_one_hot_batch, C_s_s, C_a_s)
        L_dyn = F.mse_loss(predicted_next_states, next_state_batch)

        # --- 3. Regularization Loss (L_reg) ---
        log_prob_s_s, log_prob_a_s, log_prob_s_r, log_prob_a_r = self.causal_module.get_log_probs()
        L_reg = -(self.LAMBDA_S_S * log_prob_s_s + self.LAMBDA_A_S * log_prob_a_s + \
                  self.LAMBDA_S_R * log_prob_s_r + self.LAMBDA_A_R * log_prob_a_r)
        
        total_loss = L_rew + L_dyn + L_reg
        return total_loss, L_rew, L_dyn, L_reg
    
    def save(self, save_dir):
        models = [self.reward_model, self.dynamics_model, self.causal_module]
        for model in models:
            if hasattr(model, 'save') and callable(getattr(model, 'save')):
                model.save(save_dir)