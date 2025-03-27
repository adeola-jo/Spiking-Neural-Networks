# Spiking Neural Networks for Neuromorphic Data

This project implements Spiking Neural Networks (SNNs) for neuromorphic datasets using the SNNTorch library. The code provides a complete pipeline for training, testing, and visualizing spiking neural networks.

## Project Structure

```
Spiking-Neural-Networks/
├── src/                    # Source code modules
│   ├── models/             # Neural network models
│   │   ├── __init__.py
│   │   └── snn_models.py   # SNN model implementations
│   ├── utils/              # Utility functions
│   │   ├── __init__.py
│   │   ├── checkpoint.py   # Checkpoint saving/loading utilities
│   │   └── visualization.py # Visualization utilities
│   ├── data/               # Data handling modules
│   │   ├── __init__.py
│   │   └── dataset.py      # Dataset preparation functions
│   └── __init__.py
├── scripts/                # Executable scripts
│   ├── train_nmnist.py     # Training script for N-MNIST
│   ├── visualize_nmnist.py # Visualization script for N-MNIST samples
│   └── visualize_activations.py # Activation visualization
├── checkpoints/            # Directory for model checkpoints
├── logs/                   # Directory for logs and visualizations
└── README.md               # Project documentation
```

## Key Features

- **Modular Design**: Clear separation of models, utilities, and data handling
- **Multiple SNN Architectures**: 
  - `SpikingCNN`: Convolutional SNN with Leaky Integrate-and-Fire (LIF) neurons
  - `SpikingResNet`: Residual SNN with Adaptive LIF neurons
- **Complete Training Pipeline**: Training, validation, and testing
- **Extensive Visualization Tools**: Dataset visualization, spike patterns, membrane potentials, t-SNE embeddings
- **Neuromorphic Dataset Support**: N-MNIST dataset with event-based data

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

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Spiking-Neural-Networks.git
   cd Spiking-Neural-Networks
   ```

2. Set up a virtual environment:
   ```
   python -m venv snn
   source snn/bin/activate  # On Windows: snn\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install torch torchvision tonic snntorch numpy matplotlib tqdm scikit-learn seaborn
   ```

## Quick Start

### Training a Model

To train a basic SpikingCNN model on N-MNIST:

```bash
python scripts/train_nmnist.py --model cnn --epochs 10 --batch_size 128 --time_steps 25
```

For the more advanced SpikingResNet model:

```bash
python scripts/train_nmnist.py --model resnet --epochs 15 --batch_size 64 --scheduler --visualize
```

### Visualizing the Dataset

To visualize samples from the N-MNIST dataset:

```bash
python scripts/visualize_nmnist.py --num_samples 10
```

### Visualizing Model Activations

To visualize activations and spike patterns in a trained model:

```bash
python scripts/visualize_activations.py --model cnn --visualize_all
```

## Command-Line Arguments

The training script supports many customizable parameters:

- Model parameters: `--model`, `--beta`, `--threshold`, `--alpha`, `--dropout`
- Training parameters: `--epochs`, `--batch_size`, `--time_steps`, `--lr`, `--weight_decay`, `--optimizer`, `--loss`
- Utility parameters: `--data_dir`, `--save_dir`, `--log_dir`, `--resume`, `--device`, `--test_only`, `--visualize`

## Models

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

## SNNTorch Features

This project leverages several key features of SNNTorch:

- Neuron models: `Leaky`, `ALIF`, `Synaptic`
- Surrogate gradients: `fast_sigmoid`
- Functional utilities: `reset`, `rate_coding`
- Spike visualization: `spikeplot.raster`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [SNNTorch](https://snntorch.readthedocs.io/en/latest/)
- [Tonic](https://tonic.readthedocs.io/en/latest/)
- [N-MNIST Dataset](https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full)
