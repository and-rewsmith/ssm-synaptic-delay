import torch
import torch.nn.functional as F

class SynapticDelay(torch.nn.Module):
    def __init__(self, input_dim, max_delay):
        super().__init__()
        self.max_delay = max_delay
        # Learned projection to get delay weights
        self.delay_proj = torch.nn.Linear(input_dim, max_delay)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
        Returns:
            Delayed tensor of shape [batch, seq_len, dim]
        """
        B, L, D = x.shape

        # 1. Project to get delay logits and apply softmax
        delay_logits = self.delay_proj(x)  # [B, L, max_delay]
        delay_weights = F.softmax(delay_logits, dim=-1)  # Each input distributes across delays

        # 2. Create delay matrix (parallel operation)
        indices = torch.arange(L, device=x.device).unsqueeze(0) - torch.arange(L, device=x.device).unsqueeze(1)
        delay_mask = (indices >= 0) & (indices < self.max_delay)
        
        delay_matrix = torch.zeros(B, L, L, device=x.device)
        delay_matrix[:, delay_mask] = delay_weights[:, :, indices[delay_mask]]

        # 3. Apply delays with single matrix multiply
        u_delayed = torch.bmm(delay_matrix, x)

        return u_delayed

def test_synaptic_delay():
    torch.manual_seed(42)
    B, L, D = 1, 4, 8  # batch, seq_len, dim
    
    # Create model and input
    model = SynapticDelay(input_dim=D, max_delay=3)
    x = torch.randn(B, L, D)
    
    # Forward pass
    delayed = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", delayed.shape)
    
    # Basic shape test
    assert delayed.shape == x.shape, "Output shape should match input shape"
    print("Shape test passed!")

if __name__ == "__main__":
    test_synaptic_delay()