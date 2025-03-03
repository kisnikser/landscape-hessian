import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import random
import json
import argparse
from omegaconf import OmegaConf


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


name2dataset = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
}


###############################################################################

def main(config):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the dataset
    train_dataset = name2dataset[config.dataset_name]('./data', download=True, train=True, transform=transform)
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(config.train_size))
    
    val_dataset = name2dataset[config.dataset_name]('./data', download=True, train=False, transform=transform)
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(config.val_size))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    for hidden_size in tqdm(config.hidden_size_list, desc='Hidden size loop'):
        for num_layers in tqdm(config.num_layers_list, desc='Number of layers loop'):
            
            accuracy_samples = []
            results_samples = []
            
            for sample_idx in tqdm(range(config.num_samples), desc='Sample loop', leave=False):
                
                # Initialize the network, loss function, and optimizer
                model = Net(config.input_size, hidden_size, config.output_size, num_layers).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True)
                accuracy_list = []
                
                # Train the network
                for _ in tqdm(range(config.num_epochs), desc='Train loop', leave=False):  # loop over the dataset multiple times
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
                        
                        # Validate
                        with torch.no_grad():
                            correct = 0
                            total = 0
                            for x_val, y_val in val_dataloader:
                                x_val, y_val = x_val.to(device), y_val.to(device)
                                x_val = x_val.view(-1, model.input_size)
                                outputs_val = model(x_val)
                                _, predicted = torch.max(outputs_val, 1)
                                total += y_val.size(0)
                                correct += (predicted == y_val).sum().item()
                            accuracy = correct / total
                        
                        accuracy_list.append(accuracy)
                    
                accuracy_samples.append(accuracy_list)

                # Get the optimal parameters
                theta_opt = list(model.parameters())

                # Compute the loss function values at the optimal parameters
                loss_values = []

                with torch.no_grad():
                    for x, y in tqdm(train_dataset, desc='Loss values loop', leave=False):
                        x, y = x.to(device), torch.tensor([y]).to(device)
                        x = x.view(-1, model.input_size)
                        loss = criterion_params(theta_opt, x, y, criterion, model).item()
                        loss_values.append(loss)

                results_samples.append({
                    'loss_values': loss_values,
                    'accuracy': accuracy_list
                })

            results.append({
                'h': model.hidden_size,
                'L': model.num_layers,
                'val_batch_size': config.val_batch_size,
                'samples': results_samples
            })
            
    with open(config.save_path, 'w') as f:
        json.dump(results, f, indent=4)


###############################################################################        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting loss values for FC network")
    parser.add_argument("--config_path", type=str, default='config.yml', help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)
