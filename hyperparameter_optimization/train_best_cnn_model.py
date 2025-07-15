import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import math

# Add parent directory to path to access modules from main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.generate_data_set import create_combined_dataset
from medmnist import PneumoniaMNIST
from torchvision import transforms
from data_preprocessing.data_cleaning import PreprocessTransform
from wandb_sweep import ModelWithDynamicParams


def get_data_loaders(batch_size):
    """Create data loaders for training, validation, and test"""
    transform = transforms.Compose([
        PreprocessTransform(),
    ])
    train_dataset = create_combined_dataset(
        mnist_data=PneumoniaMNIST(split='train', download=True),
        raw_data_dir='data/raw',
        augmented_data_dir='data/augmented',
        transform=transform, 
        split='train'
    )
    val_dataset = create_combined_dataset(
        mnist_data=PneumoniaMNIST(split='val', download=True),
        raw_data_dir='data/raw',
        transform=transform, 
        split='val'
    )
    test_dataset = create_combined_dataset(
        mnist_data=PneumoniaMNIST(split='test', download=True),
        # raw_data_dir='data/raw',
        transform=transform, 
        split='test'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_with_early_stopping(config, save_path):
    """Train model with early stopping, save best model, and return loss curves"""
    # Model parameters
    kernel_size = config['kernel_size']
    dropout_rate = config['dropout_rate']
    conv1_channels = config['conv1_channels']
    conv2_channels = config['conv2_channels']
    fc1_units = config['fc1_units']
    # Training parameters
    lr = config['learning_rate']
    batch_size = config['batch_size']
    optimizer_name = config['optimizer']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    early_stopping_patience = config.get('early_stopping_patience', 5)
    early_stopping_delta = config.get('early_stopping_delta', 0.0)

    train_loader, val_loader, _ = get_data_loaders(batch_size)
    model = ModelWithDynamicParams(
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        fc1_units=fc1_units
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler with warmup and cosine annealing
    warmup_epochs = 3  # Number of epochs for warmup
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = num_epochs * len(train_loader)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, math.cos(0.5 * math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Early stopping
    best_f1 = 0
    early_stopping_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step the scheduler after each optimizer step
            train_loss += loss.item() * inputs.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        # Compute F1 for early stopping
        predicted = [1 if o > 0.5 else 0 for o in all_outputs]
        f1 = f1_score(all_targets, predicted)
        if f1 > best_f1 + early_stopping_delta:
            best_f1 = f1
            early_stopping_counter = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={f1:.4f}")
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model, train_losses, val_losses


def evaluate_model(model, batch_size):
    """Evaluate model on test set and print metrics"""
    _, _, test_loader = get_data_loaders(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Test", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    predicted = [1 if o > 0.5 else 0 for o in all_outputs]
    precision = precision_score(all_targets, predicted)
    recall = recall_score(all_targets, predicted)
    f1 = f1_score(all_targets, predicted)
    accuracy = accuracy_score(all_targets, predicted)
    try:
        auc = roc_auc_score(all_targets, all_outputs)
    except ValueError:
        auc = float('nan')
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    return precision, recall, f1, accuracy, auc


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Hardcoded paths instead of argparse
    class Args:
        config = 'hyperparameter_optimization/best_model_config.yaml'  
        model_out = 'best_model.pth'
        loss_plot = 'loss_curve.png'
    args = Args()
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Train
    model, train_losses, val_losses = train_with_early_stopping(config, args.model_out)
    # Plot losses
    plot_losses(train_losses, val_losses, args.loss_plot)
    # Evaluate
    evaluate_model(model, config['batch_size'])

if __name__ == "__main__":
    main() 