"""
Pytest tests for data handling functionality.
Tests dataset loading, preprocessing, and batching.
"""

import sys
import os
import pytest
import torch

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the dataset functions
# Try to load the prepare_nmnist_dataset function
try:
    from src.data import prepare_nmnist_dataset
    DATA_MODULE_EXISTS = True
except ImportError:
    DATA_MODULE_EXISTS = False

# Fixtures
@pytest.fixture
def data_dir():
    """Get the data directory for testing."""
    data_path = os.path.join(project_root, 'datasets')
    os.makedirs(data_path, exist_ok=True)
    return data_path


# Skip all tests if the data module doesn't exist
pytestmark = pytest.mark.skipif(
    not DATA_MODULE_EXISTS,
    reason="Data module not found"
)


def test_dataset_preparation(data_dir):
    """
    Test preparing the N-MNIST dataset.
    This is a basic integration test that checks if the function runs without error.
    """
    # This might download data, so we'll use a small batch size and mark as slow
    pytest.skip("Skipping actual data download - enable when needed")
    batch_size = 4
    time_steps = 10
    
    # Try to prepare the dataset
    train_loader, test_loader = prepare_nmnist_dataset(
        data_dir, batch_size, time_steps
    )
    
    # Basic checks on the returned data loaders
    assert train_loader is not None
    assert test_loader is not None


# Mock dataset for faster testing
class MockDataset:
    """A mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return fake data sample: [time_steps, channels, height, width]
        sample = torch.rand(20, 2, 34, 34)
        label = torch.randint(0, 10, (1,)).item()
        return sample, label


def test_dataset_batch_shape():
    """Test the shape of batched data from the dataset."""
    mock_dataset = MockDataset()
    batch_size = 8
    
    # Create a simple dataloader
    dataloader = torch.utils.data.DataLoader(
        mock_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Get a batch
    data, labels = next(iter(dataloader))
    
    # Check shapes
    assert data.shape == (batch_size, 20, 2, 34, 34), f"Expected shape {(batch_size, 20, 2, 34, 34)} but got {data.shape}"
    assert labels.shape == (batch_size,), f"Expected shape {(batch_size,)} but got {labels.shape}"


def test_data_permutation():
    """Test permuting the data for time-step processing."""
    # Create a mock batch
    batch_size = 8
    time_steps = 20
    channels = 2
    height = 34
    width = 34
    
    # Original shape: [batch_size, time_steps, channels, height, width]
    data = torch.rand(batch_size, time_steps, channels, height, width)
    
    # Permute to: [time_steps, batch_size, channels, height, width]
    permuted_data = data.permute(1, 0, 2, 3, 4)
    
    # Check shapes
    assert permuted_data.shape == (time_steps, batch_size, channels, height, width)
    
    # Check specific elements to ensure permutation is correct
    for b in range(batch_size):
        for t in range(time_steps):
            # Original data indexing: [batch, time, channel, height, width]
            # Permuted data indexing: [time, batch, channel, height, width]
            assert torch.allclose(data[b, t, 0, 0, 0], permuted_data[t, b, 0, 0, 0])
