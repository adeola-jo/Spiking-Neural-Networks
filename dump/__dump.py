
# """
# Dataset preparation functions for N-MNIST
# """
# import torch
# import tonic
# import numpy as np
# import tonic.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence


# def prepare_nmnist_dataset(data_dir, batch_size, time_steps, num_workers=4):
#     """
#     Prepare the N-MNIST dataset using Tonic
    
#     Args:
#         data_dir: Directory for datasets
#         batch_size: Batch size for training/testing
#         time_steps: Number of time steps for simulation
#         num_workers: Number of workers for data loading
        
#     Returns:
#         train_loader: DataLoader for training data
#         test_loader: DataLoader for testing data
#     """
#     # Define transformations
#     frame_transform = transforms.Compose([
#         transforms.Denoise(filter_time=10000),
#         transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=time_steps)
#         # transforms.ToFrame(sensor_size=(34, 34, 2), ti_window=time_steps)
#     ])
    
#     # Load the datasets
#     train_dataset = tonic.datasets.NMNIST(
#         save_to=data_dir, 
#         train=True, 
#         transform=frame_transform, 
#         # download=True
#     )
    
#     test_dataset = tonic.datasets.NMNIST(
#         save_to=data_dir, 
#         train=False, 
#         transform=frame_transform, 
#         # download=True
#     )
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=num_workers, 
#         # collate_fn=collate_fn, 
#         pin_memory=True
#     )
    
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=num_workers,
#         # collate_fn=collate_fn, 
#         pin_memory=True
#     )
    
#     return train_loader, test_loader


# # def collate_fn(batch):
# #     """
# #     Custom collate function that handles variable-length event sequences
# #     by padding to the max length in the batch.
# #     """
# #     # Extract data and targets
# #     data, targets = zip(*batch)
    
# #     # Determine the maximum time dimension in this batch
# #     max_time = max([d.shape[0] for d in data])
    
# #     # Create a padded batch
# #     padded_data = []
# #     for i, item in enumerate(data):
# #         # Convert numpy array to torch tensor first
# #         item_tensor = torch.from_numpy(item) if isinstance(item, np.ndarray) else item
        
# #         # Get current dimensions
# #         curr_time, channels, height, width = item_tensor.shape
        
# #         # Create padded tensor with the correct dtype
# #         padded = torch.zeros((max_time, channels, height, width), 
# #                             dtype=torch.float32)  # Always use float32 for model input
        
# #         # Copy data to padded tensor (with type conversion if needed)
# #         padded[:curr_time] = item_tensor.float()  # Convert to float
# #         padded_data.append(padded)
    
# #     # Stack the padded tensors
# #     data_batch = torch.stack(padded_data)
# #     targets_batch = torch.tensor(targets, dtype=torch.long)  # Use long for class indices
    
# #     return data_batch, targets_batch