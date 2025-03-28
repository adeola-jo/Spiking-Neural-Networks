"""
Pytest tests for SNN models.
Tests the initialization and forward pass of SNN models.
"""

import sys
import os
import pytest
import torch

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.snn_models import SpikingCNN, SpikingResNet

# Fixtures
@pytest.fixture
def device():
    """Get the device to use for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_input(device):
    """Create a sample input tensor."""
    batch_size = 4
    input_channels = 2
    height = 34
    width = 34
    return torch.randn(batch_size, input_channels, height, width).to(device)

@pytest.fixture
def cnn_model(device):
    """Create a SpikingCNN model."""
    model = SpikingCNN(input_channels=2).to(device)
    return model

@pytest.fixture
def resnet_model(device):
    """Create a SpikingResNet model."""
    model = SpikingResNet(input_channels=2).to(device)
    return model


# SpikingCNN Tests
class TestSpikingCNN:
    """Tests for the SpikingCNN model."""
    
    def test_model_initialization(self, cnn_model):
        """Test that the model initializes with all required layers."""
        assert hasattr(cnn_model, 'conv1')
        assert hasattr(cnn_model, 'lif1')
        assert hasattr(cnn_model, 'conv2')
        assert hasattr(cnn_model, 'lif2')
        assert hasattr(cnn_model, 'conv3')
        assert hasattr(cnn_model, 'lif3')
        assert hasattr(cnn_model, 'fc')
        assert hasattr(cnn_model, 'lif4')
        assert hasattr(cnn_model, 'readout')
    
    def test_forward_pass(self, cnn_model, sample_input):
        """Test the forward pass and output shape."""
        output = cnn_model(sample_input)
        
        # Check output shape (batch_size, num_classes)
        assert output.shape == (sample_input.size(0), 10)
    
    def test_reset_states(self, cnn_model, sample_input):
        """Test resetting hidden states."""
        # First, do a forward pass to initialize hidden states
        cnn_model(sample_input)
        
        # Reset states
        cnn_model.reset_states()
        
        # Check if the forward pass still works after reset
        output = cnn_model(sample_input)
        assert output.shape == (sample_input.size(0), 10)


# SpikingResNet Tests
class TestSpikingResNet:
    """Tests for the SpikingResNet model."""
    
    def test_model_initialization(self, resnet_model):
        """Test that the model initializes with all required layers."""
        assert hasattr(resnet_model, 'conv1')
        assert hasattr(resnet_model, 'lif1')
        assert hasattr(resnet_model, 'res1_conv1')
        assert hasattr(resnet_model, 'res1_lif1')
        assert hasattr(resnet_model, 'res2_conv1')
        assert hasattr(resnet_model, 'res2_lif1')
        assert hasattr(resnet_model, 'fc')
        assert hasattr(resnet_model, 'fc_lif')
        assert hasattr(resnet_model, 'readout')
    
    def test_forward_pass(self, resnet_model, sample_input):
        """Test the forward pass and output shapes."""
        output, mem = resnet_model(sample_input)
        
        # Check output shapes (batch_size, num_classes)
        assert output.shape == (sample_input.size(0), 10)
        assert mem.shape == (sample_input.size(0), 10)
    
    def test_reset_states(self, resnet_model, sample_input):
        """Test resetting hidden states."""
        # First, do a forward pass to initialize hidden states
        resnet_model(sample_input)
        
        # Reset states
        resnet_model.reset_states()
        
        # Check if the forward pass still works after reset
        output, mem = resnet_model(sample_input)
        assert output.shape == (sample_input.size(0), 10)


# Integration test - run a full sequence of time steps
def test_time_sequence_processing(cnn_model, device):
    """Test processing a full sequence of time steps."""
    batch_size = 4
    time_steps = 10
    input_channels = 2
    height = 34
    width = 34
    
    # Create a sequence of inputs
    sequence = torch.randn(time_steps, batch_size, input_channels, height, width).to(device)
    
    # Reset model state
    cnn_model.reset_states()
    
    # Process each time step
    spk_rec = []
    for t in range(time_steps):
        spk_out = cnn_model(sequence[t])
        spk_rec.append(spk_out)
    
    # Stack outputs
    outputs = torch.stack(spk_rec, dim=0)
    
    # Check output shape (time_steps, batch_size, num_classes)
    assert outputs.shape == (time_steps, batch_size, 10)
    
    # Test rate coding (sum across time)
    rate_coded = outputs.sum(dim=0)
    assert rate_coded.shape == (batch_size, 10)
