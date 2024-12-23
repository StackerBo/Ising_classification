import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import os
import numpy as np
import re
import time
import logging

# check the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# 01. Prepare the dataset
class IsingDataset(Dataset):
    def __init__(self, Tc, L=10):
        self.L = L
        BASE_DIR = "Data_configurations"

        # Prepare the data list
        data_list = []
        label_list = []

        # Read the data
        T_list = np.linspace(1.0, 3.5, 1000)
        T_list = T_list[::50]
        for temp in T_list:
            file = f"L{L}_T{temp:.4f}.npy"

            # Load file
            data_np = np.load(os.path.join(BASE_DIR, f"{L}", file))
            count = 0
            for ele in data_np[-200:]:
                ele_flatten = ele.flatten()
                data_list.append(ele_flatten)
                count += 1

            # Prepare the label
            if temp < Tc:
                label_list.extend([0] * count)
            elif temp >= Tc:
                label_list.extend([1] * count)

        # Prepare the data and label tensor
        data_list = np.array(data_list)
        self.data = torch.tensor(data_list, dtype=torch.float32).to(device)
        self.label = torch.tensor(label_list, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# 02. Define the model
class FNN(nn.Module):
    def __init__(self, L=10, hidden_dim=100, output_dim=2):
        input_dim = L * L
        super(FNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
    
def run(Tc, L, model_path):
    logging.info(f"Starting training model for Tc = {Tc:.2f}...")
    # 03. Prepare the dataset and model
    logging.info("Preparing the dataset and model...")
    Ising_dataset = IsingDataset(Tc, L)
    logging.info(f"Dataset size: {len(Ising_dataset)}")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(Ising_dataset))
    val_size = len(Ising_dataset) - train_size
    train_dataset, val_dataset = random_split(Ising_dataset, [train_size, val_size])

    batch_size = 50
    num_epochs = 200

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    lr = 0.005
    weight_decay = 0
    model = FNN(L).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logging.info(f"Device: {device}")
    logging.info(f"Batch size: {batch_size}, Learning rate: {lr}, L = {L}")
    logging.info(f"Starting training for {num_epochs} epochs.")

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        start_time = time.time()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # Calculate training accuracy for this epoch
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_accuracy = val_correct / val_total
        end_time = time.time()
        logging.info(f"Epoch {epoch + 1} / {num_epochs}: Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}, Time: {end_time - start_time:.2f} sec")

    # Save the model
    torch.save(model.state_dict(), model_path)
    logging.info(f"Finished! Model is saved at {model_path}")

if __name__ == "__main__":
    L = 30
    Tc = 2.269
    run(2.269, L, f"model_regression_demo/model_{L}.pth")