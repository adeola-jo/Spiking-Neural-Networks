"""
Dataset preparation functions for N-MNIST
"""
import torch
import tonic
import numpy as np
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
    # Define transformations with fixed time steps
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        # Use fixed time_window instead of variable
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=time_steps)
    ])
    
    # Load the datasets
    train_dataset = tonic.datasets.NMNIST(
        save_to=data_dir, 
        train=True, 
        transform=frame_transform
    )
    
    test_dataset = tonic.datasets.NMNIST(
        save_to=data_dir, 
        train=False, 
        transform=frame_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, test_loader


def collate_fn(batch):
    """
    Custom collate function that handles event sequences and ensures
    we get a consistent [batch_size, time_steps, channels, height, width] format
    """
    # Extract data and targets
    data, targets = zip(*batch)
    
    # Check the shape of the data to determine how to process it
    sample_shape = data[0].shape
    
    # If time dimension is first, we need to ensure all samples have consistent time steps
    if len(sample_shape) == 4:  # (time, channels, height, width)
        # Ensure all samples have the same time dimension if using frames
        time_steps = sample_shape[0]
        batch_data = []
        
        for item in data:
            # Convert to float32 tensor if needed
            item_tensor = torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item.float()
            
            # Make sure the tensor has the right shape
            if item_tensor.shape[0] != time_steps:
                # Handle samples with different time dimensions
                # Either truncate or pad to match expected time steps
                if item_tensor.shape[0] > time_steps:
                    # Truncate
                    item_tensor = item_tensor[:time_steps]
                else:
                    # Pad with zeros
                    pad_size = time_steps - item_tensor.shape[0]
                    padding = torch.zeros((pad_size, *item_tensor.shape[1:]), dtype=torch.float32)
                    item_tensor = torch.cat([item_tensor, padding], dim=0)
            
            batch_data.append(item_tensor)
        
        # Stack all tensors into a single batch
        data_batch = torch.stack(batch_data)
        
    else:
        # Handle any other format by converting to tensor
        data_batch = torch.stack([torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d.float() for d in data])
    
    # Convert targets to tensor
    targets_batch = torch.tensor(targets, dtype=torch.long)
    
    return data_batch, targets_batch

