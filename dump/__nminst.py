import torch
from torch.utils.data import DataLoader, random_split
from snntorch.spikevision import spikedata
import matplotlib.pyplot as plt
from IPython.display import HTML
import snntorch.spikeplot as splt
# import torch.nn as nn
# import snntorch as snn
# import tonic
# import tonic.transforms as transforms
# from snntorch import surrogate
# from snntorch import utils


# Import the neuromorphic MNIST Dataset 
def download_mnist_data(data_dir='dataset/nmnist'):
    train_ds = spikedata.NMNIST(data_dir, train=True, download_and_create=True, num_steps=300, dt=1000)
    test_ds = spikedata.NMNIST(data_dir, train=False, download_and_create=True, num_steps=300, dt=1000)
    return train_ds, test_ds


def create_dataloaders(train_ds, test_ds, shuffle=True, batch_size=128, val_split=0.1):
    # Calculate split sizes
    train_size = int((1 - val_split) * len(train_ds))
    val_size = len(train_ds) - train_size
    
    # Split the training dataset
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# Create visualizers using snntorch's built-in spikeplot
def visualize_sample(dataloader, sample_idx=None, save_path=None, interval=10):
    """
    Visualize a sample from the dataset using snntorch's spikeplot
    
    Args:
        dataloader: The dataloader containing the dataset
        sample_idx: Index of the sample to visualize (random if None)
        save_path: Path to save the animation (None for no saving)
        interval: Interval between frames in milliseconds
    """
    if sample_idx is None:
        sample_idx = torch.randint(0, len(dataloader.dataset), (1,)).item()
        
    # Get the sample and label
    sample, label = dataloader.dataset[sample_idx]
    
    # Sum the on/off channels
    if sample.dim() > 2:  # If there are separate on/off channels
        combined_sample = sample[:, 0] + sample[:, 1]
    else:
        combined_sample = sample
    
    # Create the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title(f"N-MNIST Sample - Label: {label}")
    
    # Use snntorch's animator
    anim = splt.animator(combined_sample, fig, ax, interval=interval)
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=1000//interval)
    
    return anim


# Example usage:
if __name__ == "__main__":
    # Download dataset
    train_ds, test_ds = download_mnist_data()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, test_ds, batch_size=64)
    
    # Visualize a random sample
    anim = visualize_sample(train_loader, save_path="nmnist_sample.mp4", interval=10)
    
    # If in a Jupyter notebook, display the animation
    try:
        display(HTML(anim.to_html5_video()))
    except:
        print("Animation created. Run in Jupyter notebook to display.")
    
    print(f"Dataset loaded with {len(train_loader.dataset)} training, "
          f"{len(val_loader.dataset)} validation, and {len(test_loader.dataset)} test samples")