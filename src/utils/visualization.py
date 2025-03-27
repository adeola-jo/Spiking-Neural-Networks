"""
Visualization utilities for SNN models and results
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_results(train_losses, test_losses, train_accs, test_accs, save_dir):
    """
    Plot training and testing losses and accuracies
    
    Args:
        train_losses: List of training losses
        test_losses: List of testing losses
        train_accs: List of training accuracies
        test_accs: List of testing accuracies
        save_dir: Directory to save the plots
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracies')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'))
    plt.close()

def save_results(results, save_dir):
    """
    Save training results to disk
    
    Args:
        results: Dict containing training results
        save_dir: Directory to save the results
    """
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

def visualize_nmnist_sample(sample, target):
    """
    Visualize an N-MNIST sample
    
    Args:
        sample: Tensor of shape [time_steps, channels, height, width]
        target: Target label
    """
    # Sum over time dimension to visualize the accumulated events
    summed_sample = sample.sum(0).detach().cpu().numpy()
    
    plt.figure(figsize=(8, 4))
    
    # Plot positive events
    plt.subplot(1, 2, 1)
    plt.imshow(summed_sample[0], cmap='hot')
    plt.title(f'Positive Events (Label: {target})')
    plt.colorbar()
    
    # Plot negative events
    plt.subplot(1, 2, 2)
    plt.imshow(summed_sample[1], cmap='hot')
    plt.title('Negative Events')
    plt.colorbar()
    
    plt.tight_layout()
    
    return plt.gcf()

def visualize_spikes(spike_record, num_steps_to_display=25, num_neurons=10):
    """
    Visualize spike recordings
    
    Args:
        spike_record: Tensor of spike recordings [num_steps, batch_size, num_neurons]
        num_steps_to_display: Number of time steps to display
        num_neurons: Number of neurons to display
    """
    plt.figure(figsize=(12, 6))
    spike_record_np = spike_record[:num_steps_to_display, 0, :num_neurons].detach().cpu().numpy()
    
    plt.imshow(spike_record_np.T, cmap='binary', aspect='auto')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.colorbar(label='Spike')
    
    return plt.gcf()

def visualize_membrane_potential(mem_record, neuron_idx=0):
    """
    Visualize membrane potential over time
    
    Args:
        mem_record: Tensor of membrane potential recordings [num_steps, batch_size, num_neurons]
        neuron_idx: Index of the neuron to visualize
    """
    plt.figure(figsize=(12, 4))
    mem_record_np = mem_record[:, 0, neuron_idx].detach().cpu().numpy()
    
    plt.plot(mem_record_np)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time Step')
    plt.ylabel('Membrane Potential')
    plt.title(f'Membrane Potential of Neuron {neuron_idx}')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()
