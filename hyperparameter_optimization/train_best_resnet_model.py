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
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Add parent directory to path to access modules from main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.generate_data_set import create_combined_dataset
from medmnist import PneumoniaMNIST
from torchvision import transforms
from data_preprocessing.data_cleaning import PreprocessTransform

# Directory for saving best ResNet model
RESNET_SWEEP_DIR = os.path.join(os.path.dirname(__file__), 'resnet_sweep_results')
os.makedirs(RESNET_SWEEP_DIR, exist_ok=True)

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
        transform=transform, 
        split='test'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

def normalize_to_01(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def grayscale_to_rgb(tensor):
    return tensor.repeat(1, 3, 1, 1)

def freeze_layers(model, num_layers_to_unfreeze=1):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last N layers of last block
    last_stage = model.resnet.encoder.stages[-1]
    last_block = last_stage.layers[-1]
    last_conv_layers = last_block.layer[-(num_layers_to_unfreeze):]
    for conv_layer in last_conv_layers:
        for param in conv_layer.parameters():
            param.requires_grad = True
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

def train_with_early_stopping(config, save_path):
    lr = config['learning_rate']
    batch_size = config['batch_size']
    dropout_rate = config['dropout_rate']
    optimizer_name = config['optimizer']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    patience = config.get('patience', 5)
    num_layers_to_unfreeze = config.get('num_layers_to_unfreeze', 1)

    train_loader, val_loader, _ = get_data_loaders(batch_size)
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 1)
    )
    model.num_labels = 1
    freeze_layers(model, num_layers_to_unfreeze)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    best_f1 = 0
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = normalize_to_01(inputs)
            inputs = grayscale_to_rgb(inputs)
            inputs = processor(inputs, return_tensors="pt")['pixel_values'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
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
                inputs = normalize_to_01(inputs)
                inputs = grayscale_to_rgb(inputs)
                inputs = processor(inputs, return_tensors="pt")['pixel_values'].to(device)
                outputs = model(inputs).logits
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        predicted = [1 if o > 0.5 else 0 for o in all_outputs]
        f1 = f1_score(all_targets, predicted)
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={f1:.4f}")
    model.load_state_dict(torch.load(save_path))
    return model, train_losses, val_losses

def evaluate_model(model, batch_size):
    _, _, test_loader = get_data_loaders(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
    model = model.to(device)
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Test", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = normalize_to_01(inputs)
            inputs = grayscale_to_rgb(inputs)
            inputs = processor(inputs, return_tensors="pt")['pixel_values'].to(device)
            outputs = model(inputs).logits
            outputs = torch.sigmoid(outputs)
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
    class Args:
        config = 'hyperparameter_optimization/best_resnet_config.yaml' 
        model_out = os.path.join(RESNET_SWEEP_DIR, 'best_resnet_model.pth')
        loss_plot = os.path.join(RESNET_SWEEP_DIR, 'resnet_loss_curve.png')
    args = Args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model, train_losses, val_losses = train_with_early_stopping(config, args.model_out)
    plot_losses(train_losses, val_losses, args.loss_plot)
    evaluate_model(model, config['batch_size'])

if __name__ == "__main__":
    main() 