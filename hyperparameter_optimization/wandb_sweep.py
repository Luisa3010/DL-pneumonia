import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import math

# Add parent directory to path to access modules from main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.generate_data_set import create_combined_dataset
from medmnist import PneumoniaMNIST
from torchvision import transforms
from data_preprocessing.data_cleaning import PreprocessTransform

class ModelWithDynamicParams(nn.Module):
    def __init__(self, kernel_size=5, dropout_rate=0.25, conv1_channels=16, conv2_channels=32, fc1_units=128):
        super(ModelWithDynamicParams, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_channels, kernel_size=kernel_size)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=kernel_size)
        
        # Calculate the size after convolutions and pooling
        conv1_out_size = 28 - kernel_size + 1
        pool1_out_size = conv1_out_size // 2
        conv2_out_size = pool1_out_size - kernel_size + 1
        pool2_out_size = conv2_out_size // 2
        
        # Fully connected layers
        self.final_features = conv2_channels * pool2_out_size * pool2_out_size
        self.fc1 = nn.Linear(self.final_features, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 1)  # Binary classification
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        
        # Second conv block
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, self.final_features)
        
        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Sigmoid for binary classification
        return torch.sigmoid(x)

def get_data_loaders(batch_size):
    """Create data loaders for training and validation"""
    transform = transforms.Compose([
        PreprocessTransform(),
    ])
    
    # Create datasets
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def train_and_evaluate(config=None):
    """Train and evaluate a model with hyperparameters from wandb config"""
    # Initialize wandb
    with wandb.init(config=config) as run:

        # Get hyperparameters from wandb config
        config = wandb.config
        
        # Model parameters
        kernel_size = config.kernel_size
        dropout_rate = config.dropout_rate
        conv1_channels = config.conv1_channels
        conv2_channels = config.conv2_channels
        fc1_units = config.fc1_units
        
        # Training parameters
        lr = config.learning_rate
        batch_size = config.batch_size
        optimizer_name = config.optimizer
        weight_decay = config.weight_decay
        
        # Training process parameters
        num_epochs = config.num_epochs
        early_stopping_patience = config.early_stopping_patience
        early_stopping_delta = config.early_stopping_delta
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(batch_size)
        
        # Create model
        model = ModelWithDynamicParams(
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            fc1_units=fc1_units
        )
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Learning rate scheduler with warmup and cosine annealing
        warmup_epochs = 3  # Number of epochs for warmup
        warmup_steps = warmup_epochs * len(train_loader)
        total_steps = num_epochs * len(train_loader)
        
        def lr_lambda(current_step):
            # During warmup, we start from 0 and linearly increase to base_lr
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # After warmup, we do cosine decay from base_lr to 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, math.cos(0.5 * math.pi * progress))  # Cosine decay from 1 to 0
                
        # Initialize optimizer with the base learning rate from config
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping
        best_f1 = 0
        early_stopping_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                # Step the scheduler and get current learning rate
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"learning_rate": current_lr})
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            # Initialize counters for metrics
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            all_targets = []
            all_outputs = []
            
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    true_positives += ((predicted == 1) & (targets == 1)).sum().item()
                    false_positives += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
                    
                    all_outputs.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
            
            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / total
            
            # Calculate precision, recall and F1 score
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Compute ROC AUC
            try:
                roc_auc = roc_auc_score(all_targets, all_outputs)
            except ValueError:
                print("Error computing ROC AUC")
                roc_auc = float('nan')  # If only one class present in y_true, roc_auc_score is not defined
            
            # Log metrics to wandb
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "val_roc_auc": roc_auc,
                "epoch": epoch
            })
            
            # Early stopping check
            if f1 > best_f1 + early_stopping_delta:
                best_f1 = f1
                early_stopping_counter = 0
                # Save best model state
                best_model_state = model.state_dict()
                # Save model locally
                model_path = os.path.join(wandb.run.dir, 'best_model.pth')
                torch.save({
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'f1_score': best_f1,
                    'epoch': epoch,
                    'hyperparameters': {
                        'kernel_size': kernel_size,
                        'dropout_rate': dropout_rate,
                        'conv1_channels': conv1_channels,
                        'conv2_channels': conv2_channels,
                        'fc1_units': fc1_units,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'optimizer': optimizer_name,
                        'weight_decay': weight_decay,
                        'warmup_epochs': warmup_epochs
                    }
                }, model_path)
                # Save to wandb
                wandb.save(model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return best_f1

def run_wandb_sweep():
    """Run hyperparameter optimization with wandb sweep"""
    # Load sweep configuration from YAML
    sweep_config_path = os.path.join(os.path.dirname(__file__), 'wandb_sweep_config.yaml')
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="pneumonia-cnn-sweep")
    
    # Run sweep
    wandb.agent(sweep_id, function=train_and_evaluate, count=sweep_config['max_runs'])
    
    try:
        # Get entity from environment variable or default to username
        entity = os.environ.get('WANDB_ENTITY', wandb.api.default_entity)
        project = "pneumonia-cnn-sweep"
        print(f"Sweep completed. Check results at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    except:
        print("Sweep complete")

def load_best_model(model_path):
    """Load the best model from a saved checkpoint"""
    checkpoint = torch.load(model_path)
    
    # Filter only model architecture parameters
    model_params = {
        'kernel_size': checkpoint['hyperparameters']['kernel_size'],
        'dropout_rate': checkpoint['hyperparameters']['dropout_rate'],
        'conv1_channels': checkpoint['hyperparameters']['conv1_channels'],
        'conv2_channels': checkpoint['hyperparameters']['conv2_channels'],
        'fc1_units': checkpoint['hyperparameters']['fc1_units']
    }
    
    model = ModelWithDynamicParams(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['hyperparameters']

if __name__ == "__main__":
    # Initialize wandb
    wandb.login()
    run_wandb_sweep() 