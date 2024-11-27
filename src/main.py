import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from pscan import pscan
from synaptic_delay import compute_synaptic_delay


# Constants
NUM_SEQUENCES = 1
SEQUENCE_LENGTH = 200
NUM_MODES = 1
FREQ_RANGE = (1.5, 10.5)
AMP_RANGE = (0.5, 1.5)
PHASE_RANGE = (0, 2 * np.pi)
NUM_BINS = 25
BATCH_SIZE = NUM_SEQUENCES
HIDDEN_DIM = 64
CHANNELS = NUM_BINS
TIMESTEPS = SEQUENCE_LENGTH
LEARNING_RATE = 1e-2
NUM_EPOCHS = 200
OUTPUT_DIM = NUM_BINS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DelayedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DelayedMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.mlp = nn.Sequential(
            DelayedLinear(input_size, hidden_size),
            DelayedLinear(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size),
        )

    def init_buffer(self, batch_size: int):
        for module in self.mlp:
            if isinstance(module, DelayedLinear):
                module.init_buffer(batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DelayedLinear(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DelayedLinear, self).__init__()
        self.input_size = input_size
        self.delay_gate_input = nn.Linear(input_size, input_size)
        self.delay_gate_buffer = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

    def init_buffer(self, batch_size: int):
        self.buffer = torch.zeros(batch_size, self.input_size).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.size()
        # Use a local variable for the buffer
        outputs = []

        for t in range(seq_len):
            current_input = x[:, t, :]

            # compute what of the current input is getting delayed
            decay_weights = self.sigmoid(self.delay_gate_input(current_input))
            immediate_contribution = current_input * decay_weights
            delayed_contribution = (1 - decay_weights) * current_input

            # add delayed portion to buffer
            #
            # notably, this happens before the buffer release so some input can
            # be delayed, interact with buffer, and be released, all in the same
            # timestep
            self.buffer = self.buffer + delayed_contribution

            # determine how much of the buffer to release
            buffer_decay_weights = self.sigmoid(self.delay_gate_buffer(self.buffer))
            buffer_release = self.buffer * buffer_decay_weights
            self.buffer = self.buffer * (1 - buffer_decay_weights)

            # combine the immediate and delayed contributions and feed through MLP
            combined_input = immediate_contribution + buffer_release
            output = self.linear(combined_input)
            outputs.append(output)

        final_output = torch.stack(outputs, dim=1)
        return final_output

def initialize_wandb():
    wandb.init(project="ssm-synaptic-delay", config={
        "num_sequences": NUM_SEQUENCES,
        "sequence_length": SEQUENCE_LENGTH,
        "num_modes": NUM_MODES,
        "freq_range": FREQ_RANGE,
        "amp_range": AMP_RANGE,
        "phase_range": PHASE_RANGE,
        "num_bins": NUM_BINS,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "channels": CHANNELS,
        "timesteps": TIMESTEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
    })

class SSMSynapticDelay(nn.Module):
    def __init__(self):
        super(SSMSynapticDelay, self).__init__()

        self.multilayer = nn.Sequential(
            SSMSynapticDelayLayer(CHANNELS, HIDDEN_DIM),
            SSMSynapticDelayLayer(HIDDEN_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.multilayer(x)

class SSMSynapticDelayLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SSMSynapticDelayLayer, self).__init__()
        self.delay_proj = nn.Linear(input_dim, TIMESTEPS)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.B = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Reshape input to (TIMESTEPS, BATCH_SIZE, CHANNELS)
        x = x.transpose(0, 1)
        
        x_delayed = compute_synaptic_delay(x, self.delay_proj)
        u = self.U(x_delayed)
        u = u.transpose(0, 1)
        b = torch.sigmoid(self.B(x_delayed))
        b = b.transpose(0, 1)
        
        pscan_output = pscan(self.A.weight, b, u, torch.zeros(x.size(1), HIDDEN_DIM).to(DEVICE))
        hidden = pscan_output.transpose(0, 1)
        
        return hidden

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

def train_model(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss, num_epochs: int) -> None:
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            model.init_buffer(BATCH_SIZE)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, NUM_BINS), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

def inference_and_plot(model, inputs):
    model.init_buffer(BATCH_SIZE)
    model.eval()
    with torch.no_grad():
        predictions = []
        teacher_forcing = True

        for t in range(inputs.size(1)):
            if t == inputs.size(1) // 2:
                teacher_forcing = False
                print("\n*** Switching from teacher forcing to autoregressive prediction ***\n")

            if teacher_forcing:
                print(inputs.shape)
                current_input = inputs[:, t, :]
                print(current_input.shape)
            else:
                # Use the last prediction for autoregressive inference
                if predictions:
                    last_output = predictions[-1]
                    probabilities = torch.softmax(last_output, dim=-1)
                    predicted_indices = probabilities.argmax(dim=-1)
                    current_input = torch.zeros_like(inputs[:, t, :])
                    # current_input[:, -1, :].scatter_(1, predicted_indices.unsqueeze(-1), 1)
                    current_input.scatter_(1, predicted_indices.unsqueeze(-1), 1)
                else:
                    current_input = inputs[:, t, :]

            output = model(current_input.unsqueeze(1))
            predictions.append(output[:, -1, :])

        predictions = torch.stack(predictions, dim=1)
        original_data = inputs[0].cpu().numpy()
        predicted_data = predictions[0].argmax(dim=-1).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(original_data)), original_data.argmax(axis=-1), label="Original Data", color="blue")
    plt.plot(range(len(predicted_data)), predicted_data, 'o', label="Predicted Output", color="orange")
    plt.axvline(x=inputs.size(1) // 2, color='grey', linestyle='--', label='Teacher Forcing Ends')
    plt.xlabel("Timestep")
    plt.ylabel("Bin Index")
    plt.title("Inference: Teacher-Forced vs. Autoregressive Prediction")
    plt.legend()
    plt.savefig("inference.png")

def main():
    print("Running on:", DEVICE)

    initialize_wandb()

    # Generate and prepare data
    continuous_waveforms = generate_waveforms(NUM_SEQUENCES, SEQUENCE_LENGTH, NUM_MODES,
                                           FREQ_RANGE, AMP_RANGE, PHASE_RANGE)
    one_hot_waveforms = discretize_waveforms(continuous_waveforms, NUM_BINS)

    inputs = torch.tensor(one_hot_waveforms, dtype=torch.float32).to(DEVICE)
    targets = torch.tensor(np.argmax(one_hot_waveforms, axis=-1), dtype=torch.long).to(DEVICE)
    targets = targets[:, 1:]
    inputs = inputs[:, :-1]

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and training components
    # model = SSMSynapticDelay().to(DEVICE)
    model = DelayedMLP(input_size=NUM_BINS, hidden_size=HIDDEN_DIM, output_size=OUTPUT_DIM).to(DEVICE)
    model.init_buffer(BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, data_loader, optimizer, criterion, NUM_EPOCHS)

    # Run inference and plot results
    for inputs, _ in data_loader:
        inputs = inputs.to(DEVICE)
        inference_and_plot(model, inputs)
        break

    wandb.finish()

if __name__ == "__main__":
    main()