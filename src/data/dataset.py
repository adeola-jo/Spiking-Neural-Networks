"""
Dataset preparation functions for N-MNIST
"""
import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

def prepare_nmnist_dataset(data_dir, batch_size, time_steps, num_workers=4):
    """
    Prepare the N-MNIST dataset using Tonic
    
    Args:
        data_dir: Directory for datasets
        batch_size: Batch size for training/testing
        time_steps: Number of time steps for simulation
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
    """
    # Define transformations
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=(2, 34, 34), time_window=time_steps)
    ])
    
    # Load the datasets
    train_dataset = tonic.datasets.NMNIST(
        root=data_dir, 
        train=True, 
        transform=frame_transform, 
        download=True
    )
    
    test_dataset = tonic.datasets.NMNIST(
        root=data_dir, 
        train=False, 
        transform=frame_transform, 
        download=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, test_loader
