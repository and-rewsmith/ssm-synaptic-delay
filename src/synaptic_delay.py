import torch
import torch.nn.functional as F

def compute_synaptic_delay(x, delay_proj):
    """
    Args:
        x: Input tensor of shape [batch, seq_len, dim]
        delay_proj: Linear projection layer for computing delays
    Returns:
        Delayed tensor of shape [batch, seq_len, dim]
    """
    B, L, D = x.shape
    max_delay = delay_proj.out_features

    # 1. Project to get delay logits and apply softmax
    delay_logits = delay_proj(x)  # [B, L, max_delay]
    delay_weights = F.softmax(delay_logits, dim=-1)  # [B, L, max_delay]

    # 2. Create delay matrix
    src = torch.arange(L, device=x.device)
    tgt = torch.arange(L, device=x.device)
    diff = tgt.unsqueeze(1) - src.unsqueeze(0)  # [L, L]
    mask = (diff >= 0) & (diff < max_delay)  # [L, L]
    
    delay_matrix = torch.zeros(B, L, L, device=x.device)
    # Create positional indices
    i = torch.arange(L, device=x.device)  # [L]
    j = torch.arange(L, device=x.device)  # [L]

    # Create matrix of i,j positions
    i = i.view(L, 1).expand(L, L)  # [L, L]
    j = j.view(1, L).expand(L, L)  # [L, L]

    # Calculate delays
    delays = i - j  # [L, L]
    mask = (delays >= 0) & (delays < max_delay)  # [L, L]

    # Get all valid delays
    delay_matrix = torch.zeros(B, L, L, device=x.device)
    delay_matrix[:, mask] = delay_weights[:, j[mask], delays[mask]]

    # 3. Apply delays
    u_delayed = torch.bmm(delay_matrix, x)
    
    return u_delayed

def test_synaptic_delay():
    torch.manual_seed(42)
    B, L, D = 2, 4, 8
    max_delay = 3
    
    x = torch.randn(B, L, D)
    delay_proj = torch.nn.Linear(D, max_delay)
    
    delayed = compute_synaptic_delay(x, delay_proj)
    print("Input shape:", x.shape)
    print("Output shape:", delayed.shape)
    print("Test passed!")

if __name__ == "__main__":
    test_synaptic_delay()