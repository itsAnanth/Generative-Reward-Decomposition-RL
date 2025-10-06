import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """Predicts the unobserved Markovian reward r_t."""
    def __init__(self, state_dim, action_dim, HIDDEN_DIM):
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
    
    def save(self, save_dir):
        """Save reward model parameters."""
        torch.save(self.state_dict(), f"{save_dir}/reward.pth")