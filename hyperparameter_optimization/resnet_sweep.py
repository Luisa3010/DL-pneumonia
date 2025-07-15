import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModelForImageClassification
import math

# Add parent directory to path to access modules from main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.generate_data_set import create_combined_dataset
from medmnist import PneumoniaMNIST
from torchvision import transforms
from data_preprocessing.data_cleaning import PreprocessTransform

# Create directory for ResNet sweep results
RESNET_SWEEP_DIR = os.path.join(os.path.dirname(__file__), 'resnet_sweep_results')
os.makedirs(RESNET_SWEEP_DIR, exist_ok=True)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def normalize_to_01(tensor):
    """Normalize tensor values to [0,1] range"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def grayscale_to_rgb(tensor):
    """Convert grayscale tensor to RGB by repeating the channel"""
    # tensor shape: [batch_size, channels, height, width]
    return tensor.repeat(1, 3, 1, 1)  # Repeat the channel dimension 3 times

def freeze_layers(model, num_layers_to_unfreeze = 1):
    """
    Freeze most of the ResNet backbone, only keeping the final classification layers trainable.
    For the HuggingFace ResNetForImageClassification model, we access the backbone through
    model.resnet and freeze all layers except the last convolution layer of the last ResNet block
    and the classification head.
    
    Args:
        model: The ResNet model from HuggingFace transformers
    """
    # Print model structure for debugging
    print("\nModel structure:")
    for name, _ in model.named_children():
        print(f"Main component: {name}")
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the last stage and its last layer
    last_stage = model.resnet.encoder.stages[-1]
    last_block = last_stage.layers[-1]

    last_conv_layers = last_block.layer[-(num_layers_to_unfreeze):]  
    for conv_layer in last_conv_layers:
        for param in conv_layer.parameters():
            param.requires_grad = True
        print(f"Unfreezing layer {conv_layer} of the last ResNet block")
    
    # Unfreeze the classifier
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True
        print(f"Unfreezing classifier parameter: {name}")
    
    # Print which parameters are trainable for verification
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

def train_and_evaluate(config=None):
    """Train and evaluate a model with hyperparameters from wandb config"""
    # Initialize wandb
    with wandb.init(config=config) as run:
        # Get hyperparameters from wandb config
        config = wandb.config
        
        # Training parameters
        lr = config.learning_rate
        print(f"Learning rate: {lr}")
        batch_size = config.batch_size
        dropout_rate = config.dropout_rate
        optimizer_name = config.optimizer
        weight_decay = config.weight_decay
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(batch_size)
        
        # Load pretrained model and processor
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        
        # Modify the classifier head for binary classification with dropout
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1)
        )
        model.num_labels = 1
        
        # Freeze most layers and only train classifier and last residual block
        freeze_layers(model, config.num_layers_to_unfreeze)
        # Print trainable parameters info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"\nModel Parameter Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
        print(f"\nTrainable layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Calculate the ratio of negative to positive samples in your training set
        num_positives = 12470
        num_negatives = 4520
        pos_weight = torch.tensor([num_negatives / num_positives], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer selection
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Early stopping variables
        best_val_loss = float('inf')  # Track the lowest val_loss
        epochs_no_improve = 0
        patience = config.patience

        # Learning rate scheduler with warmup and cosine annealing
        warmup_epochs = 3  # Number of epochs for warmup
        warmup_steps = warmup_epochs * len(train_loader)
        total_steps = config.num_epochs * len(train_loader)

        def lr_lambda(current_step):
            # During warmup, we start from 0 and linearly increase to base_lr
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # After warmup, we do cosine decay from base_lr to 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, math.cos(0.5 * math.pi * progress))  # Cosine decay from 1 to 0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        for epoch in range(config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Normalize inputs to [0,1] range before processing
                # inputs = normalize_to_01(inputs)
                
                # Convert grayscale to RGB
                inputs = grayscale_to_rgb(inputs)
                
                # Process inputs through the processor
                inputs = processor(inputs, return_tensors="pt")["pixel_values"].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs).logits
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"learning_rate": current_lr})
                
                train_loss += loss.item() * inputs.size(0)
            
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
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Normalize inputs to [0,1] range before processing
                    # inputs = normalize_to_01(inputs)
                    
                    # Convert grayscale to RGB
                    inputs = grayscale_to_rgb(inputs)
                    
                    # Process inputs through the processor
                    inputs = processor(inputs, return_tensors="pt")["pixel_values"].to(device)
                    
                    outputs = model(inputs).logits
                    loss = criterion(outputs, targets)
                    
                    # Apply sigmoid for metrics and thresholding
                    outputs_prob = torch.sigmoid(outputs)
                    predicted = (outputs_prob > 0.5).float()
                    
                    val_loss += loss.item() * inputs.size(0)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    true_positives += ((predicted == 1) & (targets == 1)).sum().item()
                    false_positives += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
                    all_outputs.extend(outputs_prob.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
            
            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / total
            
            # Debug print block for metric diagnosis (no numpy)
            print("\n[DEBUG] Validation targets and outputs:")
            print(f"  all_targets length: {len(all_targets)}")
            print(f"  all_outputs length: {len(all_outputs)}")
            print(f"  Unique values in all_targets: {set(all_targets)}")
            if all_outputs:
                min_out = min(all_outputs)
                max_out = max(all_outputs)
                mean_out = sum(all_outputs) / len(all_outputs)
                print(f"  Min/Max/Mean of all_outputs: {min_out:.4f} / {max_out:.4f} / {mean_out:.4f}")
            print(f"  Sample all_targets: {all_targets[:10]}")
            print(f"  Sample all_outputs: {[float(x) for x in all_outputs[:10]]}")

            # Calculate precision, recall and F1 score
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Compute ROC AUC
            try:
                roc_auc = roc_auc_score(all_targets, all_outputs)
            except ValueError:
                print("Error computing ROC AUC")
                roc_auc = float('nan')
            
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
            
            # Save best model and early stopping logic (now based on val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                wandb.run.summary["best_val_loss"] = val_loss
                # Save model locally in the ResNet sweep directory
                model_path = os.path.join(RESNET_SWEEP_DIR, f'best_model_run_{wandb.run.id}.pth')
                # torch.save({
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'val_loss': val_loss,
                #     'epoch': epoch,
                #     'hyperparameters': {
                #         'learning_rate': lr,
                #         'batch_size': batch_size,
                #         'dropout_rate': dropout_rate,
                #         'optimizer': optimizer_name,
                #         'weight_decay': weight_decay,
                #         'num_layers_to_unfreeze': config.num_layers_to_unfreeze
                #     }
                # }, model_path)
                # Save to wandb
                wandb.save(model_path)
                print(f"New best val_loss: {val_loss:.4f} at epoch {epoch+1}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs")
            
            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break
        
        return best_val_loss  # Optionally return best_val_loss instead of best_f1

if __name__ == "__main__":
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  
        'metric': {
            'name': 'val_f1',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2,
            },
            'batch_size': {
                'value': 32
            },
            'num_epochs': {
                'value': 30
            },
            'patience': {
                'value': 5  # Stop if no improvement for 5 epochs
            },
            'num_layers_to_unfreeze': {
                'values': [1,2,3]
            },
            'optimizer': {
                'values': ['Adam', 'SGD', 'rmsprop']
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-2,
            },
            'dropout_rate': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.5,
            }
        }
    }
    
    # Initialize the sweep with a different project name
    sweep_id = wandb.sweep(sweep_config, project="pneumonia-resnet-sweep")
    
    # Run the sweep
    wandb.agent(sweep_id, function=train_and_evaluate, count=30)  # Run 30 trials 