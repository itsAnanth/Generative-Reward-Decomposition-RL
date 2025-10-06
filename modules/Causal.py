import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalModel(nn.Module):
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
    
    def save(self, save_dir):
        """Save causal model parameters."""
        torch.save(self.state_dict(), f"{save_dir}/causal.pth")