import torch
from torch import nn
from pscan import pscan
from synaptic_delay import compute_synaptic_delay
import wandb
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
HIDDEN_DIM = 2
CHANNELS = 3
TIMESTEPS = 4

class SSMSynapticDelay(nn.Module):
    def __init__(self):
        super(SSMSynapticDelay, self).__init__()

        self.delay_proj = nn.Linear(CHANNELS, TIMESTEPS)
        self.U = nn.Linear(CHANNELS, HIDDEN_DIM)
        self.A = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False) # TODO: we can use bias if we want
        self.B = nn.Linear(CHANNELS, HIDDEN_DIM)

    def forward(self, x):
        assert x.shape == (TIMESTEPS, BATCH_SIZE, CHANNELS)

        x_delayed = compute_synaptic_delay(x, self.delay_proj)
        assert x_delayed.shape == (TIMESTEPS, BATCH_SIZE, CHANNELS)
        u = self.U(x)
        u = u.transpose(0, 1)

        b = torch.sigmoid(self.B(x))
        b = b.transpose(0, 1)
        assert b.shape == (BATCH_SIZE, TIMESTEPS, HIDDEN_DIM)

        pscan_output = pscan(self.A.weight, b, u, torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(DEVICE))
        return pscan_output.transpose(0, 1)
        
def generate_waveforms(num_sequences: int, sequence_length: int, num_modes: int,
                       freq_range: tuple, amp_range: tuple, phase_range: tuple) -> np.ndarray:
    waveforms = np.zeros((num_sequences, sequence_length))
    t = np.linspace(0, 2 * np.pi, sequence_length, endpoint=False)
    for i in range(num_sequences):
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            waveforms[i] += amplitude * np.sin(frequency * t + phase)
    return waveforms


def discretize_waveforms(waveforms: np.ndarray, num_bins: int) -> np.ndarray:
    min_val, max_val = waveforms.min(), waveforms.max()
    scaled_waveforms = (waveforms - min_val) / (max_val - min_val) * (num_bins - 1)
    discretized_waveforms = np.clip(np.round(scaled_waveforms), 0, num_bins - 1).astype(int)
    one_hot_waveforms = np.eye(num_bins)[discretized_waveforms]
    return one_hot_waveforms

def initialize_wandb():
    wandb.init(project="ssm-synaptic-delay", config={
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "channels": CHANNELS,
        "timesteps": TIMESTEPS,
    })

if __name__ == "__main__":
    initialize_wandb()

    x = torch.ones(TIMESTEPS, BATCH_SIZE, CHANNELS).to(DEVICE)
    model = SSMSynapticDelay().to(DEVICE)
    out = model.forward(x)
    print(out.shape)

    # this is dumb we shouldn't do it
    # a_t = sigma(Q * x_t)
    # b_t = S * b_{t-1} + (a_t) * u_t
    # c_t = sigma(Q * x_t + R * b_t)
    # z_t = A * z_t + c_t * b_t

    # instead we will do some simple synaptic delay
    # x_delayed = compute_delays(x)  # our delay matrix approach
    # z_t = A * z_t + B * x_delayed
    # NOTE: B can be time varying!

    # then we need to convert to mamba
    # h_t = exp(δ_t * A) * h_{t-1} + δ_t * B * u_delayed_t