# Hyperparameter Optimization for Pneumonia CNN

This directory contains tools for optimizing the hyperparameters of the CNN model used for pneumonia detection.

## Files

- `wandb_sweep_config.yaml`: Configuration file defining the hyperparameters to optimize, their ranges, and optimization settings.
- `wandb_sweep.py`: Python script that implements the hyperparameter optimization process using Weights & Biases (wandb) sweep.

## Requirements

To run the hyperparameter optimization, you need to install the following dependencies:

```bash
pip install wandb pyyaml matplotlib scikit-learn tqdm
```

## How to Run

1. Make sure your model training dependencies are already installed
2. Ensure your data preprocessing pipeline is working correctly
3. Login to wandb (you'll need to create an account at wandb.ai if you don't have one):
   ```bash
   wandb login
   ```
4. Run the optimization script:
   ```bash
   python wandb_sweep.py
   ```

## Output

The optimization process will produce the following outputs:

1. Terminal output showing the progress of each trial
2. A wandb dashboard showing:
   - Real-time visualization of training metrics
   - Hyperparameter relationships and importance
   - Model performance comparisons
   - Resource usage statistics
3. Best model checkpoints saved to wandb artifacts

## Configuring the Optimization

You can modify the `wandb_sweep_config.yaml` file to:

1. Change the hyperparameter search spaces
2. Adjust the number of trials
3. Modify early stopping criteria
4. Change the optimization metric (currently set to F1 score)
5. Adjust resource allocation (CPUs, GPUs)

## Using the Best Model

After optimization, you can use the best model from the wandb dashboard:

1. Go to your wandb project
2. Find the best run based on the F1 score
3. Download the model checkpoint from the artifacts section
4. Load the model using the provided `load_best_model()` function:

```python
from wandb_sweep import load_best_model

# Load the best model
model, hyperparameters = load_best_model('path_to_best_model.pth')
```

The saved model checkpoint contains all necessary information to reproduce the training setup, including:
- Model architecture parameters
- Training hyperparameters
- Optimizer state
- Performance metrics 