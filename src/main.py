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
BATCH_SIZE = 1
HIDDEN_DIM = 8
CHANNELS = NUM_BINS
TIMESTEPS = SEQUENCE_LENGTH
LEARNING_RATE = 1e-2
NUM_EPOCHS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.delay_proj = nn.Linear(CHANNELS, TIMESTEPS)
        self.U = nn.Linear(CHANNELS, HIDDEN_DIM)
        self.A = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.B = nn.Linear(CHANNELS, HIDDEN_DIM)
        self.output_layer = nn.Linear(HIDDEN_DIM, CHANNELS)

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
        
        # Project to output space
        output = self.output_layer(hidden)
        return output.transpose(0, 1)  # Return to (BATCH_SIZE, TIMESTEPS, CHANNELS)

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
    model.eval()
    with torch.no_grad():
        predictions = []
        teacher_forcing = True

        for t in range(inputs.size(1)):
            if t == inputs.size(1) // 2:
                teacher_forcing = False
                print("\n*** Switching from teacher forcing to autoregressive prediction ***\n")

            if teacher_forcing:
                current_input = inputs[:, :t+1, :]
            else:
                # Use the last prediction for autoregressive inference
                if predictions:
                    last_output = predictions[-1]
                    probabilities = torch.softmax(last_output, dim=-1)
                    predicted_indices = probabilities.argmax(dim=-1)
                    current_input = torch.zeros_like(inputs[:, :t+1, :])
                    current_input[:, -1, :].scatter_(1, predicted_indices.unsqueeze(-1), 1)
                else:
                    current_input = inputs[:, :t+1, :]

            output = model(current_input)
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
    model = SSMSynapticDelay().to(DEVICE)
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