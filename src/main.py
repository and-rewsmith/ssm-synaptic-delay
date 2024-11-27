import torch
from torch import nn

from pscan import pscan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMESTEPS = 10
BATCH_SIZE = 2
CHANNELS = 3
HIDDEN_DIM = 5

class SSMSynapticDelay(nn.Module):
    def __init__(self):
        super(SSMSynapticDelay, self).__init__()
        self.Q = nn.Linear(CHANNELS, HIDDEN_DIM)
        self.R = nn.Linear(CHANNELS, HIDDEN_DIM)
        self.A = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x):
        assert x.shape == (TIMESTEPS, BATCH_SIZE, CHANNELS)
        a = torch.sigmoid(self.Q(x))



if __name__ == "__main__":
    print("---")

    x = torch.ones(TIMESTEPS, BATCH_SIZE, CHANNELS).to(DEVICE)
    inital_states = torch.zeros(BATCH_SIZE, CHANNELS).to(DEVICE)

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