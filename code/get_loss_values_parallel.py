import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random
import json
import argparse
from omegaconf import OmegaConf
import multiprocessing as mp
from tqdm import tqdm

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

def criterion_params(params, x, y, criterion, model):
    names = list(n for n, _ in model.named_parameters())
    output = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, x)
    loss = criterion(output, y)
    return loss

def process_h_l(h, L, config, device, train_dataset, train_dataloader, val_dataloader):
    
    accuracy_samples = []
    results_samples = []
    
    for sample_idx in range(config.num_samples):
        
        # Initialize the network, loss function, and optimizer
        model = Net(config.input_size, h, config.output_size, L).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        accuracy_list = []
        
        # Train the network
        for _ in range(config.num_epochs):
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
            for x, y in train_dataset:
                x, y = x.to(device), torch.tensor([y]).to(device)
                x = x.view(-1, model.input_size)
                loss = criterion_params(theta_opt, x, y, criterion, model).item()
                loss_values.append(loss)

        results_samples.append({
            'loss_values': loss_values,
            'accuracy': accuracy_list
        })

    return {
        'h': h,
        'L': L,
        'val_batch_size': config.val_batch_size,
        'samples': results_samples
    }

def worker(gpu_id, config_dict, task_queue, result_queue):
    config = OmegaConf.create(config_dict)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = getattr(datasets, config.dataset_name)(
        './data', download=True, train=True, transform=transform)
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(config.train_size))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, 
        shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = getattr(datasets, config.dataset_name)(
        './data', download=True, train=False, transform=transform)
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(config.val_size))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.val_batch_size, 
        shuffle=False, num_workers=4, pin_memory=True)
    
    while True:
        task = task_queue.get()
        if task is None:
            break
        h, L = task
        result = process_h_l(h, L, config, device, train_dataset, train_dataloader, val_dataloader)
        result_queue.put(result)
    
    result_queue.put(None)

def main(config):
    tasks = [(h, L) for h in config.hidden_size_list for L in config.num_layers_list]
    num_gpus = torch.cuda.device_count()
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for task in tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)
    
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker,
            args=(gpu_id, OmegaConf.to_container(config), task_queue, result_queue)
        )
        p.start()
        workers.append(p)
    
    results = []
    completed_workers = 0
    with tqdm(total=len(tasks), desc='Processing tasks') as pbar:
        while completed_workers < num_gpus:
            result = result_queue.get()
            if result is None:
                completed_workers += 1
            else:
                results.append(result)
                pbar.update(1)
    
    for p in workers:
        p.join()
    
    with open(config.save_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='config.yml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)