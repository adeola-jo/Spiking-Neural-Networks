"""
Pytest tests for utility functions.
Tests checkpointing, visualization, and other utility functions.
"""

import sys
import os
import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import pickle

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils import save_checkpoint, load_checkpoint, plot_results, save_results
    from src.utils.visualization import plot_snn_spikes, visualize_nmnist_sample
    UTILS_MODULE_EXISTS = True
except ImportError:
    UTILS_MODULE_EXISTS = False


# Skip all tests if the utils module doesn't exist
pytestmark = pytest.mark.skipif(
    not UTILS_MODULE_EXISTS,
    reason="Utils module not found"
)


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 10, kernel_size=3)
            self.fc = torch.nn.Linear(10, 5)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)
    
    return MockModel()


@pytest.fixture
def mock_optimizer(mock_model):
    """Create a mock optimizer for testing."""
    return torch.optim.Adam(mock_model.parameters(), lr=0.001)


@pytest.fixture
def mock_training_history():
    """Create mock training history for testing."""
    return {
        'train_losses': [2.5, 2.0, 1.5, 1.0, 0.8],
        'test_losses': [2.7, 2.1, 1.7, 1.2, 1.0],
        'train_accs': [40.0, 50.0, 60.0, 70.0, 75.0],
        'test_accs': [38.0, 45.0, 55.0, 65.0, 68.0],
    }


@pytest.fixture
def mock_spike_tensor():
    """Create a mock spike tensor for testing."""
    batch_size = 4
    time_steps = 20
    neurons = 10
    
    # Create random binary spike tensor
    spikes = torch.rand(time_steps, batch_size, neurons) > 0.7
    return spikes.float()


@pytest.fixture
def mock_nmnist_sample():
    """Create a mock N-MNIST sample for testing."""
    time_steps = 10
    channels = 2
    height = 34
    width = 34
    
    # Create a random sample with binary events
    sample = torch.rand(time_steps, channels, height, width) > 0.9
    return sample.float()


# Test checkpoint functions
def test_save_and_load_checkpoint(temp_dir, mock_model, mock_optimizer):
    """Test saving and loading checkpoints."""
    # Create checkpoint data
    checkpoint_data = {
        'epoch': 5,
        'state_dict': mock_model.state_dict(),
        'optimizer': mock_optimizer.state_dict(),
        'best_acc': 85.5,
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
    save_checkpoint(checkpoint_data, False, temp_dir, filename='checkpoint.pth')
    
    # Check that file exists
    assert os.path.exists(checkpoint_path)
    
    # Try to load checkpoint
    loaded_model = type(mock_model)()
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.001)
    
    checkpoint = load_checkpoint(temp_dir, loaded_model, loaded_optimizer, filename='checkpoint.pth')
    
    # Check that the checkpoint loaded correctly
    assert checkpoint['epoch'] == 5
    assert checkpoint['best_acc'] == pytest.approx(85.5)
    
    # Check model state dict
    for key in mock_model.state_dict().keys():
        assert torch.allclose(
            checkpoint['state_dict'][key], 
            mock_model.state_dict()[key]
        )


# Test visualization functions
def test_plot_results(temp_dir, mock_training_history):
    """Test plotting training results."""
    # Call plot function
    plot_results(
        mock_training_history['train_losses'],
        mock_training_history['test_losses'],
        mock_training_history['train_accs'],
        mock_training_history['test_accs'],
        temp_dir
    )
    
    # Check that the plot file was created
    plot_path = os.path.join(temp_dir, 'training_results.png')
    assert os.path.exists(plot_path)


def test_save_results(temp_dir, mock_training_history):
    """Test saving training results."""
    # Add some extra data
    results = {
        **mock_training_history,
        'best_acc': 75.0,
        'args': {'model': 'cnn', 'epochs': 10}
    }
    
    # Save results
    save_results(results, temp_dir)
    
    # Check that the file exists
    results_path = os.path.join(temp_dir, 'results.pkl')
    assert os.path.exists(results_path)
    
    # Load the results and check
    with open(results_path, 'rb') as f:
        loaded_results = pickle.load(f)
    
    assert loaded_results['best_acc'] == 75.0
    assert loaded_results['args']['model'] == 'cnn'
    assert loaded_results['train_losses'] == mock_training_history['train_losses']


def test_plot_snn_spikes(temp_dir, mock_spike_tensor):
    """Test plotting SNN spikes."""
    # Set the output directory
    os.environ['MPLCONFIGDIR'] = temp_dir
    
    # Plot spikes
    title = "Test Spikes"
    time_steps = mock_spike_tensor.shape[0]
    fig = plot_snn_spikes(mock_spike_tensor, title, time_steps)
    
    # Check that the figure was created
    assert isinstance(fig, plt.Figure)
    
    # Close the figure to avoid warnings
    plt.close(fig)


def test_visualize_nmnist_sample(mock_nmnist_sample):
    """Test visualizing an N-MNIST sample."""
    # Visualize sample
    target = 5
    fig = visualize_nmnist_sample(mock_nmnist_sample, target)
    
    # Check that the figure was created
    assert isinstance(fig, plt.Figure)
    
    # Close the figure to avoid warnings
    plt.close(fig)
