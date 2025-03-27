"""
Models for Spiking Neural Networks using SNNTorch
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikegen
from snntorch import utils

class SpikingCNN(nn.Module):
    """
    Spiking Convolutional Neural Network for N-MNIST classification using SNNTorch
    
    Architecture:
    - 2D Convolutional layers for feature extraction
    - Leaky Integrate-and-Fire (LIF) neurons for spiking behavior
    - Readout layer for classification
    """
    def __init__(self, input_channels=2, num_time_bins=25, beta=0.5, threshold=1.0, dropout_p=0.2):
        super(SpikingCNN, self).__init__()
        
        # Spike gradient surrogate function
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate output size after convolutions and pooling
        # For N-MNIST: Input 34x34 -> 17x17 -> 8x8 -> 4x4
        feature_size = 64 * 4 * 4
        
        # Fully connected and readout layer
        self.fc = nn.Linear(feature_size, 256)
        self.dropout = nn.Dropout(dropout_p)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        
        self.readout = nn.Linear(256, 10)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        
        # Store time bins for initialization
        self.num_time_bins = num_time_bins
    
    def forward(self, x):
        """
        Forward pass for a single time step
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            spk_out: Output spikes
            mem_out: Output membrane potential
        """
        # First convolutional block
        cur = self.conv1(x)
        spk1, mem1 = self.lif1(cur)
        x = self.pool1(spk1)
        
        # Second convolutional block
        cur = self.conv2(x)
        spk2, mem2 = self.lif2(cur)
        x = self.pool2(spk2)
        
        # Third convolutional block
        cur = self.conv3(x)
        spk3, mem3 = self.lif3(cur)
        x = self.pool3(spk3)
        
        # Flatten for fully connected
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        cur = self.fc(x)
        cur = self.dropout(cur)
        spk4, mem4 = self.lif4(cur)
        
        # Readout layer
        cur = self.readout(spk4)
        spk_out, mem_out = self.lif_out(cur)
        
        return spk_out, mem_out

    def reset_states(self):
        """Reset all hidden states to initial values"""
        self.lif1.reset_()
        self.lif2.reset_()
        self.lif3.reset_()
        self.lif4.reset_()
        self.lif_out.reset_()


class SpikingResNet(nn.Module):
    """
    Spiking Residual Network for N-MNIST classification using SNNTorch
    This more advanced model includes residual connections for better performance
    """
    def __init__(self, input_channels=2, num_time_bins=25, beta=0.5, threshold=1.0, alpha=0.9, dropout_p=0.3):
        super(SpikingResNet, self).__init__()
        
        # Spike gradient surrogate function
        spike_grad = surrogate.fast_sigmoid(slope=75)  # Using steeper slope for better gradient propagation
        
        # Initial layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Using ALIF (Adaptive LIF) neurons for adaptive thresholds
        self.lif1 = snn.ALIF(
            alpha=alpha,               # Threshold decay rate
            beta=beta,                 # Membrane decay rate
            threshold=threshold,       # Firing threshold
            spike_grad=spike_grad,     # Surrogate gradient
            init_hidden=True           # Initialize hidden states
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual blocks
        self.res1_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(32)
        self.res1_lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        self.res1_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(32)
        self.res1_lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        
        self.res2_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.res2_bn1 = nn.BatchNorm2d(64)
        self.res2_lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        self.res2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res2_bn2 = nn.BatchNorm2d(64)
        self.res2_lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsample for residual connection
        self.downsample = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.down_bn = nn.BatchNorm2d(64)
        self.down_lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        
        # For N-MNIST: Input 34x34 -> 17x17 -> 8x8
        feature_size = 64 * 8 * 8
        
        # Fully connected and readout layer
        self.fc = nn.Linear(feature_size, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_p)
        # Using Synaptic Layer for improved temporal information processing
        self.fc_lif = snn.Synaptic(
            alpha=0.8,               # Synaptic decay rate
            beta=beta,               # Membrane decay rate
            threshold=threshold,     # Firing threshold
            spike_grad=spike_grad,   # Surrogate gradient
            init_hidden=True         # Initialize hidden states
        )
        
        self.readout = nn.Linear(256, 10)
        self.out_lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True)
        
        # Store time bins
        self.num_time_bins = num_time_bins
    
    def forward(self, x):
        """
        Forward pass for a single time step with residual connections
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            spk_out: Output spikes
            mem_out: Output membrane potential
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        spk, mem = self.lif1(x)
        x = self.pool1(spk)
        
        # First residual block
        residual = x
        x = self.res1_conv1(x)
        x = self.res1_bn1(x)
        spk, mem = self.res1_lif1(x)
        x = self.res1_conv2(spk)
        x = self.res1_bn2(x)
        spk, mem = self.res1_lif2(x)
        x = spk + residual  # Residual connection
        
        # Second residual block with downsampling
        residual = self.downsample(x)
        residual = self.down_bn(residual)
        residual, _ = self.down_lif(residual)
        
        x = self.res2_conv1(x)
        x = self.res2_bn1(x)
        spk, mem = self.res2_lif1(x)
        x = self.res2_conv2(spk)
        x = self.res2_bn2(x)
        spk, mem = self.res2_lif2(x)
        x = spk + residual  # Residual connection
        
        x = self.pool2(x)
        
        # Flatten for fully connected
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        x = self.fc_bn(x)
        x = self.dropout(x)
        spk, mem = self.fc_lif(x)
        
        # Readout layer
        x = self.readout(spk)
        spk_out, mem_out = self.out_lif(x)
        
        return spk_out, mem_out
    
    def reset_states(self):
        """Reset all hidden states for a new sequence"""
        self.lif1.reset_()
        self.res1_lif1.reset_()
        self.res1_lif2.reset_()
        self.res2_lif1.reset_()
        self.res2_lif2.reset_()
        self.down_lif.reset_()
        self.fc_lif.reset_()
        self.out_lif.reset_()
