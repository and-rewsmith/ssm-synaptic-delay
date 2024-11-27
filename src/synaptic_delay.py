import torch
import torch.nn.functional as F

def compute_synaptic_delay(x, delay_proj, max_delay=None):
    """
    Args:
        x: Input tensor of shape [batch, seq_len, dim]
        delay_proj: Linear projection layer for computing delays
    Returns:
        Delayed tensor of shape [batch, seq_len, dim]
    """
    # print("x")
    # print(x)
    # print()
    B, L, D = x.shape

    if max_delay is None:
        max_delay = delay_proj.out_features

    # 1. Project to get delay logits and apply softmax
    delay_logits = delay_proj(x)  # [B, L, max_delay]
    delay_weights = F.softmax(delay_logits, dim=-1)  # [B, L, max_delay]
    # print("delay weights")
    # print(delay_weights)
    # print()
    
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
    # print()
    # print(delay_weights)
    # print(delay_weights.shape)
    delay_matrix[:, mask] = delay_weights[:, j[mask], delays[mask]]
    # print(delay_matrix)
    # print(delay_matrix.shape)

    # 3. Apply delays
    u_delayed = torch.bmm(delay_matrix, x)
    # print(u_delayed)
    
    return u_delayed

def test_synaptic_delay():
    torch.manual_seed(42)
    B, L, D = 1, 3, 2
    max_delay = 3
    
    x = torch.randn(B, L, D)
    delay_proj = torch.nn.Linear(D, max_delay)
    
    delayed = compute_synaptic_delay(x, delay_proj)
    print()
    print("Input shape:", x.shape)
    print("Output shape:", delayed.shape)
    print("Test passed!")

if __name__ == "__main__":
    test_synaptic_delay()