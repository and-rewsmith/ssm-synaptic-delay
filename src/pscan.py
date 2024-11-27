import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np

def pscan(A, B, u, Y_init):
    """
    A: [state, state]  # state transition matrix
    B: [batch, seq_len, state]  # time-varying input coefficients
    u: [batch, seq_len, input_dim]  # input sequence
    Y_init: [batch, state]  # initial state
    """
    batch_size, seq_len, input_dim = u.shape
    state_size, _ = A.shape
    
    # First compute B_t * u_t for each timestep
    X = B * u  # element-wise multiplication [batch, seq_len, state]
    
    # Now following the log-space trick structure
    Y_init = Y_init[:, :, None]  # [batch, state, 1]
    Xa = torch.concat([Y_init, torch.transpose(X, 1, 2)], dim=-1)  # [batch, state, seq_len+1]
    
    X_real = torch.abs(Xa).log()
    X_complex = (Xa < 0).to(A.dtype)
    X_ = torch.complex(X_real, X_complex * torch.pi)  # [batch, state, seq_len+1]
    
    # Fix the diagonal extraction to maintain [batch, state] ordering
    A_diag = A.diagonal().unsqueeze(0)  # [1, state]
    A = A_diag.expand(batch_size, -1)  # [batch, state]
    A = A.unsqueeze(-1)  # [batch, state, 1]
    
    A_real = torch.abs(A).log()
    A_complex = (A < 0).to(A.dtype)
    A_ = torch.complex(A_real, A_complex * torch.pi)  # [batch, state, 1]
    
    A_ = A_.expand(-1, -1, seq_len + 1)  # [batch, state, seq_len+1]
    
    a_star = F.pad(torch.cumsum(A_, dim=-1), (1, 0))  # [batch, state, seq_len+2]
    a_star = a_star[..., :-1]  # Remove extra padding to match X_ size
    
    log_x0_plus_b_star = torch.logcumsumexp(X_ - a_star, dim=-1)
    log_x = a_star + log_x0_plus_b_star
    
    return torch.transpose(torch.exp(log_x).real[..., 1:], 1, 2)

def sequential_scan(A, B, u, Y_init):
    """
    Sequential implementation for correctness checking
    
    A: [state, state]  # state transition matrix
    B: [batch, seq_len, state]  # time-varying input coefficients
    u: [batch, seq_len, input_dim]  # input sequence
    Y_init: [batch, state]  # initial state
    """
    batch_size, seq_len, input_dim = u.shape
    state_size, _ = A.shape
    Y = torch.zeros(batch_size, seq_len, state_size).to(u.device)
    h = Y_init
    
    # Extract diagonal elements from A for element-wise multiplication
    A_diag = A.diagonal().unsqueeze(0)  # [1, state]
    
    for t in range(seq_len):
        h = A_diag * h + B[:, t, :] * u[:, t, :]
        Y[:, t, :] = h
    
    return Y

def test_correctness():
    torch.manual_seed(42)  # For reproducibility
    batch_size, seq_len, state_size = 2, 10, 3
    input_dim = state_size  # Making input_dim same as state_size for simplicity
    
    # Create test inputs
    A = torch.randn(state_size, state_size).to(torch.float32)
    B = torch.randn(batch_size, seq_len, state_size).to(torch.float32)
    u = torch.randn(batch_size, seq_len, input_dim).to(torch.float32)
    Y_init = torch.randn(batch_size, state_size).to(torch.float32)
    
    # Compute both versions
    result_pscan = pscan(A, B, u, Y_init)
    result_sequential = sequential_scan(A, B, u, Y_init)
    
    # Print shapes and a few values for debugging
    print("pscan shape:", result_pscan.shape)
    print("sequential shape:", result_sequential.shape)
    print("First few values pscan:", result_pscan[0, 0, :3])
    print("First few values sequential:", result_sequential[0, 0, :3])
    
    # Assert they're close
    assert torch.allclose(result_pscan, result_sequential, rtol=1e-4, atol=1e-4), \
           "Results don't match!"
    print("Correctness test passed!")

def benchmark_timing():
    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    times_pscan = []
    times_sequential = []
    
    batch_size = 32
    state_size = 16
    input_dim = state_size  # Making input_dim same as state_size for simplicity
    
    for seq_len in sizes:
        # Setup inputs
        A = torch.randn(state_size, state_size).cuda()
        B = torch.randn(batch_size, seq_len, state_size).cuda()
        u = torch.randn(batch_size, seq_len, input_dim).cuda()
        Y_init = torch.randn(batch_size, state_size).cuda()
        
        # Time pscan
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):  # Average over 100 runs
            _ = pscan(A, B, u, Y_init)
        torch.cuda.synchronize()
        times_pscan.append((time.perf_counter() - start) / 100)
        
        # Time sequential
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):  # Average over 100 runs
            _ = sequential_scan(A, B, u, Y_init)
        torch.cuda.synchronize()
        times_sequential.append((time.perf_counter() - start) / 100)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_pscan, 'o-', label='Log-space Parallel Scan')
    plt.plot(sizes, times_sequential, 'o-', label='Sequential Scan')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Scan Implementation Timing Comparison (Time-varying B)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("timing_varying_b.png")
    plt.close()

if __name__ == "__main__":
    test_correctness()
    benchmark_timing()