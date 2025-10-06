import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):
    """Predicts the next state s_{t+1}."""
    def __init__(self, state_dim, action_dim, HIDDEN_DIM):
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
    
    def save(self, save_dir):
        """Save dynamics model parameters."""
        torch.save(self.state_dict(), f"{save_dir}/dynamics.pth")