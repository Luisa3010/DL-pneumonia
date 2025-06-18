import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.basic_cnn import BasicCNN
import matplotlib.pyplot as plt
import pickle
from data_preprocessing.generate_data_set import create_combined_dataset
from medmnist import PneumoniaMNIST
from torchvision import transforms
from data_preprocessing.data_cleaning import PreprocessTransform

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Training loop for a PyTorch model
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        num_epochs: Number of training epochs
        device: Device to run training on (cuda/cpu)
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Progress bar for training batches
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, targets in train_bar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
  
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize counters for metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # No gradient calculation during validation
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy and other metrics for binary classification
                predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Calculate additional metrics (TP, FP, TN, FN for precision, recall, F1)
                true_positives += ((predicted == 1) & (targets == 1)).sum().item()
                false_positives += ((predicted == 1) & (targets == 0)).sum().item()
                false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
                
                val_bar.set_postfix(loss=loss.item())
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        
        # Calculate precision, recall and F1 score using accumulated values
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'] = history.get('val_precision', []) + [precision]
        history['val_recall'] = history.get('val_recall', []) + [recall]
        history['val_f1'] = history.get('val_f1', []) + [f1]
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, "
              f"F1: {f1:.4f}")
    
    return model, history

# TODO: Alter this to make it flexible for different models and move to a more appropriate location
if __name__ == "__main__":

    model = BasicCNN()
    
    # Create a transform pipeline that includes our preprocessing
    transform = transforms.Compose([
        PreprocessTransform(),
        # Add any additional transforms here if needed
    ])
    # Create train, validation, and test datasets
    print("Creating training dataset...")
    train_dataset = create_combined_dataset(
        mnist_data=PneumoniaMNIST(split='train', download=True),
        raw_data_dir='data/raw',
        augmented_data_dir='data/augmented',
        transform=transform, 
        split='train'
    )
    print(f"Training dataset created with {len(train_dataset)} samples")
    
    print("Creating validation dataset...")
    val_dataset = create_combined_dataset(
        mnist_data=PneumoniaMNIST(split='val', download=True),
        raw_data_dir='data/raw',
        transform=transform, 
        split='val'
    )
    print(f"Validation dataset created with {len(val_dataset)} samples")
    
    print("Creating test dataset...")
    test_dataset = create_combined_dataset(
        mnist_data=PneumoniaMNIST(split='test', download=True),
        # raw_data_dir='data/raw',
        transform=transform, 
        split='test'
    )
    print(f"Test dataset created with {len(test_dataset)} samples")


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    trained_model, history = train(model, train_loader, val_loader, 
                                  criterion, optimizer, num_epochs=10, device=device)
    

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_model.pth')
    print("Model saved to 'trained_model.pth'")
    
    # Save the training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved to 'training_history.pkl'")
    

    # Plot loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0, max(max(history['train_loss']), max(history['val_loss'])) * 1.1])  # Set y-limit with 10% margin
    plt.legend()
    
    # Plot accuracy curve
    plt.subplot(1, 4, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])  # Fixed range for accuracy (0-100%)
    plt.legend()

    # Plot precision curve
    plt.subplot(1, 4, 3)
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.ylim([0, 1.0])  # Fixed range for precision (0-100%)
    plt.legend()

    # Plot recall curve
    plt.subplot(1, 4, 4)
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.ylim([0, 1.0])  # Fixed range for recall (0-100%)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to 'training_curves.png'")