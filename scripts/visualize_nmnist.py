"""
Script to visualize the N-MNIST dataset
"""
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import snntorch as snn
from snntorch import spikeplot

def parse_args():
    parser = argparse.ArgumentParser(description='N-MNIST dataset visualization')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Directory for datasets')
    parser.add_argument('--time_steps', type=int, default=25, help='Number of time steps for simulation')
    parser.add_argument('--save_dir', type=str, default='./visualizations', help='Directory for saving visualizations')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    return parser.parse_args()

def visualize_sample(event_sample, frame_sample, target, idx, save_dir):
    """
    Visualize a sample from the N-MNIST dataset, showing both the raw events and the frames
    
    Args:
        event_sample: Raw event data
        frame_sample: Frame-transformed data
        target: Class label
        idx: Sample index
        save_dir: Directory to save visualizations
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot raw events in 3D
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    events = event_sample.numpy()
    if len(events) > 0:  # Check if there are events in the sample
        # Extract x, y, t coordinates and polarities
        xs, ys, ts, ps = events['x'], events['y'], events['t'], events['p']
        
        # Normalize time to [0, 1]
        ts_normalized = (ts - ts.min()) / (ts.max() - ts.min() if ts.max() > ts.min() else 1)
        
        # Plot positive and negative events with different colors
        ax1.scatter(xs[ps==1], ys[ps==1], ts_normalized[ps==1], c='r', alpha=0.5, s=1, label='Positive')
        ax1.scatter(xs[ps==0], ys[ps==0], ts_normalized[ps==0], c='b', alpha=0.5, s=1, label='Negative')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Time')
        ax1.set_title(f'Raw Events (Label: {target})')
        ax1.legend()
    else:
        ax1.set_title('No events in this sample')
    
    # Plot frame representation (summed over time)
    ax2 = fig.add_subplot(2, 2, 2)
    # Sum over time dimension
    frame_sum = frame_sample.sum(dim=0).numpy()
    
    # Plot positive events channel
    ax2.imshow(frame_sum[0], cmap='hot', interpolation='nearest')
    ax2.set_title('Positive Events (Summed over time)')
    
    # Plot negative events channel
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(frame_sum[1], cmap='hot', interpolation='nearest')
    ax3.set_title('Negative Events (Summed over time)')
    
    # Plot frame sequence as animation frames (using SNNTorch's animation utility)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Use SNNTorch's spikeplot to create a spike raster plot
    spike_data = frame_sample.view(frame_sample.shape[0], -1).t()  # Reshape to [pixels, time]
    spike_data_binary = (spike_data > 0).float()  # Convert to binary spikes
    
    # Plot spike raster
    spikeplot.raster(spike_data_binary[:100], ax=ax4, s=1, c='black')
    ax4.set_title('Spike Raster Plot (First 100 Pixels)')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Pixel Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_{idx}_label_{target}.png'))
    plt.close(fig)

def main():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Define transformations for event data and frame data
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=(2, 34, 34), time_window=args.time_steps)
    ])
    
    # Create dataset instances
    event_dataset = tonic.datasets.NMNIST(
        root=args.data_dir, 
        train=True, 
        transform=None,  # No transform for raw events
        download=True
    )
    
    frame_dataset = tonic.datasets.NMNIST(
        root=args.data_dir, 
        train=True, 
        transform=frame_transform,
        download=True
    )
    
    # Visualize random samples
    indices = np.random.randint(0, len(event_dataset), args.num_samples)
    
    for i, idx in enumerate(tqdm(indices, desc="Visualizing samples")):
        event_sample, target = event_dataset[idx]
        frame_sample, _ = frame_dataset[idx]
        
        visualize_sample(event_sample, frame_sample, target, i, args.save_dir)
    
    print(f"Visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()
