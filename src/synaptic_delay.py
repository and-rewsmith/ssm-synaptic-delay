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
    for b in range(B):
        for i in range(L):
            for j in range(max(0, i-max_delay+1), i+1):
                delay_matrix[b, i, j] = delay_weights[b, j, i-j]

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