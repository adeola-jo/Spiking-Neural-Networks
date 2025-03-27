# Spiking Neural Network for N-MNIST Classification

This project implements Spiking Neural Networks (SNNs) for the Neuromorphic-MNIST (N-MNIST) dataset using the SNNTorch library. The code provides a complete pipeline for training, testing, and visualizing spiking neural networks.

## Project Structure

- `nmnist.py`: Main script for handling N-MNIST dataset with basic functionality
- `models.py`: Implementation of SNN models (SpikingCNN and SpikingResNet)
- `train_nmnist.py`: Complete training pipeline with various configurable options
- `utils.py`: Utility functions for model checkpointing, visualization, and evaluation
- `visualize_nmnist.py`: Script for visualizing N-MNIST dataset samples
- `visualize_activations.py`: Script for visualizing network activations and spike patterns

## Dependencies

- PyTorch
- SNNTorch
- Tonic
- NumPy
- Matplotlib
- Tqdm
- Scikit-learn (for t-SNE visualization)
- Seaborn (for visualization)

## Installation

1. Set up a virtual environment:
   ```
   python -m venv snn
   source snn/bin/activate  # On Windows: snn\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install torch torchvision tonic snntorch numpy matplotlib tqdm scikit-learn seaborn
   ```

## Dataset

The N-MNIST dataset is a neuromorphic version of the popular MNIST dataset, recorded with event-based cameras. Each sample consists of spike events representing the dynamic changes as a digit is moved in front of the camera.

The dataset will be automatically downloaded when running the scripts.

## Usage

### Training a Model

To train a basic SpikingCNN model:

```bash
python train_nmnist.py --model cnn --epochs 10 --batch_size 128 --time_steps 25
```

For the more advanced SpikingResNet model:

```bash
python train_nmnist.py --model resnet --epochs 15 --batch_size 64 --time_steps 25 --scheduler --visualize
```

### Customizable Parameters

- Model parameters:
  - `--model`: Model architecture (`cnn` or `resnet`)
  - `--beta`: Beta parameter for LIF neurons (membrane decay factor)
  - `--threshold`: Firing threshold for LIF neurons
  - `--alpha`: Alpha parameter for adaptive neurons (threshold decay factor)
  - `--dropout`: Dropout probability

- Training parameters:
  - `--epochs`: Number of training epochs
  - `--batch_size`: Batch size for training
  - `--time_steps`: Number of time steps for simulation
  - `--lr`: Learning rate
  - `--weight_decay`: Weight decay for regularization
  - `--optimizer`: Optimizer (`adam`, `adamw`, or `sgd`)
  - `--loss`: Loss function (`ce`, `mse`, or `count`)
  - `--scheduler`: Use learning rate scheduler

- Utility parameters:
  - `--data_dir`: Directory for datasets
  - `--save_dir`: Directory for saving checkpoints
  - `--log_dir`: Directory for saving logs
  - `--resume`: Resume training from checkpoint
  - `--device`: Device to use (`cuda` or `cpu`)
  - `--test_only`: Only run inference on test set
  - `--visualize`: Visualize training results

### Evaluating a Trained Model

To evaluate a trained model:

```bash
python train_nmnist.py --model cnn --test_only --visualize
```

### Visualizing Dataset Samples

To visualize samples from the N-MNIST dataset:

```bash
python visualize_nmnist.py --num_samples 10
```

### Visualizing Network Activations

To visualize activations and spike patterns in a trained model:

```bash
python visualize_activations.py --model cnn --visualize_all
```

## Model Architectures

### SpikingCNN

A convolutional neural network with leaky integrate-and-fire (LIF) neurons:

- 3 convolutional layers with LIF neurons and max pooling
- Fully connected layer with LIF neurons
- Output layer with LIF neurons

### SpikingResNet

A more advanced residual network with adaptive LIF neurons:

- Initial convolutional layer with ALIF neurons
- 2 residual blocks with LIF neurons
- Synaptic layer for improved temporal information processing
- Output layer with LIF neurons

## SNNTorch Features Used

- Neuron models: `Leaky`, `ALIF`, `Synaptic`
- Surrogate gradients: `fast_sigmoid`
- Functional utilities: `reset`, `rate_coding`
- Spike visualization: `spikeplot.raster`

## Visualizations

The training process creates several visualizations:

- Training and testing loss/accuracy curves
- Confusion matrix (in test mode)
- Sample visualizations
- Network filter visualizations (for ResNet)

The activation visualization script generates:

- Spike raster plots for each layer
- Membrane potential plots
- t-SNE visualization of network features

## Examples

![Training Results](./logs/training_results.png)
![Spike Pattern](./activations/spikes/spike_pattern_lif_out.png)
![t-SNE Features](./activations/tsne/tsne_features.png)

## Model Performance

The SpikingCNN model typically achieves around 90-92% accuracy on the N-MNIST test set after 10 epochs of training. The SpikingResNet model can achieve slightly higher accuracy (92-95%) with the same number of epochs.

## References

- [SNNTorch Documentation](https://snntorch.readthedocs.io/en/latest/)
- [Tonic Documentation](https://tonic.readthedocs.io/en/latest/)
- [N-MNIST Dataset Paper](https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full)
