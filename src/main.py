import torch
from torch import nn

from pscan import pscan
from synaptic_delay import compute_synaptic_delay

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
HIDDEN_DIM = 2
CHANNELS = 3
TIMESTEPS = 4

class SSMSynapticDelay(nn.Module):
    def __init__(self):
        super(SSMSynapticDelay, self).__init__()

        self.delay_proj = nn.Linear(CHANNELS, TIMESTEPS)
        self.A = nn.Linear(CHANNELS, HIDDEN_DIM, bias=False)
        self.B = nn.Linear(CHANNELS, HIDDEN_DIM, bias=False)

    def forward(self, x):
        assert x.shape == (TIMESTEPS, BATCH_SIZE, CHANNELS)

        x_delayed = compute_synaptic_delay(x, self.delay_proj)
        assert x_delayed.shape == (TIMESTEPS, BATCH_SIZE, CHANNELS)

        b = self.B.weight.unsqueeze(0).repeat(TIMESTEPS, 1, 1).transpose(0, 1)
        assert b.shape == (TIMESTEPS, HIDDEN_DIM, CHANNELS)

        pscan_output = pscan(self.A.weight, b, x_delayed, torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(DEVICE))
        



if __name__ == "__main__":
    print("---")

    x = torch.ones(TIMESTEPS, BATCH_SIZE, CHANNELS).to(DEVICE)
    model = SSMSynapticDelay().to(DEVICE)
    model.forward(x)

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