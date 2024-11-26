import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMESTEPS = 10
BATCH_SIZE = 2
CHANNELS = 3

class SSMSynapticDelay(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        assert x.shape == (TIMESTEPS, BATCH_SIZE, CHANNELS)
        pass

if __name__ == "__main__":
    x = torch.ones(TIMESTEPS, BATCH_SIZE, CHANNELS).to(DEVICE)
    inital_states = torch.zeros(BATCH_SIZE, CHANNELS).to(DEVICE)

    # a_t = sigma(Q * x_t)
    # b_t = a_t * b_{t-1} + (1 - a_t) * u_t
    # c_t = sigma(Q * x_t + R * b_t)
    # z_t = A * z_t + c_t 


