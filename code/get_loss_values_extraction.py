import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import pipeline
from tqdm import tqdm
import numpy as np
import random
import json
import argparse
from omegaconf import OmegaConf
from functools import partial

# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_size=784, hidden_size=16, output_size=10, num_layers=3):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.ReLU()))
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.ReLU()))
        self.layers.append(nn.Linear(hidden_size, output_size, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Define loss function
def criterion_params(params, x, y, criterion, model):
    names = list(n for n, _ in model.named_parameters())
    output = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, x)
    loss = criterion(output, y)
    return loss

def get_loss_abs_differences(loss_values):
    loss_cumsum_values = np.cumsum(random.sample(loss_values, len(loss_values)))
    loss_mean_values = loss_cumsum_values / np.arange(1, len(loss_cumsum_values) + 1)
    loss_abs_differences = abs(np.diff(loss_mean_values))
    return loss_abs_differences

# calculate the EMA (exponential moving average)
def calculate_ema(data, window=10):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(data, weights, mode='full')[:len(data)]
    ema[:window] = ema[window]
    return ema

name2dataset = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
}

###############################################################################

def extract_features(dataset, pipe, device):
    features = []
    labels = []

    for img, label in tqdm(dataset, desc='Extracting features'):
        feature = torch.tensor(pipe(img)).to(device)
        features.append(feature)
        labels.append(label)

    features = torch.stack(features)
    labels = torch.tensor(labels)

    return features, labels

def main(config):
    # Download and load the dataset
    train_dataset = name2dataset[config.dataset_name](f'~/.pytorch/{config.dataset_name}_data/', download=True, train=True)
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(config.train_size))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", device=device, pool=True)

    # Extract features for the entire dataset
    features, labels = extract_features(train_dataset, pipe, device)

    # Create a new dataset with the extracted features
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    feature_dataset = FeatureDataset(features, labels)
    train_dataloader = torch.utils.data.DataLoader(feature_dataset, batch_size=config.train_batch_size, shuffle=True)

    results = []

    for hidden_size in tqdm(config.hidden_size_list, desc='Hidden size loop'):
        for num_layers in tqdm(config.num_layers_list, desc='Number of layers loop'):

            # Initialize the network, loss function, and optimizer
            model = Net(config.input_size, hidden_size, config.output_size, num_layers).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

            # Train the network
            for epoch in tqdm(range(config.num_epochs), desc='Train loop', leave=False):  # loop over the dataset multiple times
                for x, y in train_dataloader:
                    x, y = x.to(device), y.to(device)
                    # get the inputs
                    x = x.view(-1, model.input_size)  # flatten the input data
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

            # Get the optimal parameters
            theta_opt = list(model.parameters())
            num_param = sum(p.numel() for p in model.parameters())

            # Compute the loss function values at the optimal parameters
            loss_values = []

            with torch.no_grad():
                for x, y in tqdm(feature_dataset, desc='Loss values loop', leave=False):
                    x, y = x.to(device), torch.tensor([y]).to(device)
                    x = x.view(-1, model.input_size)
                    loss = criterion_params(theta_opt, x, y, criterion, model).item()
                    loss_values.append(loss)

            results.append({'h': model.hidden_size, 'L': model.num_layers, 'loss_values': loss_values})

    with open(config.save_path, 'w') as f:
        json.dump(results, f, indent=4)

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting loss values for FC network")
    parser.add_argument("--config_path", type=str, default='config.yml', help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)
