"""
Script to visualize activations in a trained SNN model
"""
import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import spikeplot
from snntorch import functional as SF
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from models import SpikingCNN, SpikingResNet
from utils import load_checkpoint
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='SNN Activation Visualization')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'], help='Model architecture')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory with model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./activations', help='Directory for saving visualizations')
    parser.add_argument('--time_steps', type=int, default=25, help='Number of time steps for simulation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for LIF neurons')
    parser.add_argument('--threshold', type=float, default=1.0, help='Firing threshold for LIF neurons')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--visualize_all', action='store_true', help='Visualize all layers (otherwise just output layer)')
    return parser.parse_args()

def get_model(args, device):
    """Load the appropriate SNN model based on arguments"""
    if args.model == 'cnn':
        model = SpikingCNN(
            input_channels=2,
            num_time_bins=args.time_steps,
            beta=args.beta,
            threshold=args.threshold
        ).to(device)
    else:
        model = SpikingResNet(
            input_channels=2,
            num_time_bins=args.time_steps,
            beta=args.beta,
            threshold=args.threshold
        ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint_dir, model, filename='model_best.pth.tar')
    if checkpoint:
        print(f"Loaded checkpoint with accuracy: {checkpoint['best_acc']:.2f}%")
    else:
        print("Warning: No checkpoint loaded. Using untrained model.")
    
    return model

def prepare_dataset(args):
    """Prepare the N-MNIST test dataset"""
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=(2, 34, 34), time_window=args.time_steps)
    ])
    
    test_dataset = tonic.datasets.NMNIST(
        root=args.data_dir,
        train=False,
        transform=frame_transform,
        download=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader

def hook_fn(module, input, output, activations, name):
    """Hook function to capture activations"""
    activations[name] = output.detach()

def register_hooks(model, activations, visualize_all=False):
    """Register forward hooks to capture activations"""
    hooks = []
    
    # Register hooks for specific layers based on model type
    if isinstance(model, SpikingCNN):
        if visualize_all:
            # Register hooks for all LIF layers
            hooks.append(model.lif1.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'lif1')))
            hooks.append(model.lif2.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'lif2')))
            hooks.append(model.lif3.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'lif3')))
            hooks.append(model.lif4.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'lif4')))
        # Always register hook for output layer
        hooks.append(model.lif_out.register_forward_hook(
            lambda m, i, o: hook_fn(m, i, o, activations, 'lif_out')))
    else:  # ResNet model
        if visualize_all:
            # Register hooks for selected layers
            hooks.append(model.lif1.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'lif1')))
            hooks.append(model.res1_lif2.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'res1_lif2')))
            hooks.append(model.res2_lif2.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'res2_lif2')))
            hooks.append(model.fc_lif.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, activations, 'fc_lif')))
        # Always register hook for output layer
        hooks.append(model.out_lif.register_forward_hook(
            lambda m, i, o: hook_fn(m, i, o, activations, 'out_lif')))
    
    return hooks

def visualize_spike_pattern(spike_data, title, output_path):
    """Visualize spike pattern using SNNTorch's spike plot"""
    plt.figure(figsize=(12, 8))
    
    # Get spike data (first element of tuple is spike data)
    if isinstance(spike_data, tuple):
        spike_data = spike_data[0]
    
    # If 4D tensor from conv layer (time, batch, channels, height, width), 
    # reshape to 3D (time, batch, features)
    if len(spike_data.shape) == 5:
        n_steps, batch_size, channels, height, width = spike_data.shape
        spike_data = spike_data.reshape(n_steps, batch_size, -1)
    
    # If 3D tensor (time, batch, features), get first batch
    if len(spike_data.shape) == 3:
        spike_data = spike_data[:, 0, :]  # (time, features)
    
    # Convert to binary for better visualization
    binary_spikes = (spike_data > 0).float().cpu()
    
    # Transpose to get (features, time)
    spike_data_t = binary_spikes.t()
    
    # If too many features, sample a subset
    if spike_data_t.shape[0] > 100:
        indices = np.random.choice(spike_data_t.shape[0], 100, replace=False)
        spike_data_t = spike_data_t[indices]
    
    # Plot spike raster
    ax = plt.gca()
    spikeplot.raster(spike_data_t, ax=ax)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_membrane_potential(mem_data, title, output_path):
    """Visualize membrane potential over time"""
    plt.figure(figsize=(12, 8))
    
    # Get membrane data (second element of tuple is membrane data)
    if isinstance(mem_data, tuple):
        mem_data = mem_data[1]
    
    # If 4D tensor from conv layer (time, batch, channels, height, width), 
    # reshape to 3D (time, batch, features)
    if len(mem_data.shape) == 5:
        n_steps, batch_size, channels, height, width = mem_data.shape
        mem_data = mem_data.reshape(n_steps, batch_size, -1)
    
    # If 3D tensor (time, batch, features), get first batch and sample neurons
    if len(mem_data.shape) == 3:
        mem_data = mem_data[:, 0, :]  # (time, features)
    
    # Sample neurons if too many
    if mem_data.shape[1] > 10:
        indices = np.random.choice(mem_data.shape[1], 10, replace=False)
        mem_data = mem_data[:, indices]
    
    # Plot membrane potential
    mem_data_np = mem_data.cpu().numpy()
    for i in range(mem_data_np.shape[1]):
        plt.plot(mem_data_np[:, i], label=f'Neuron {i}')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Membrane Potential')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def collect_features_for_tsne(model, dataloader, device, time_steps):
    """Collect final layer features for t-SNE visualization"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc='Collecting features'):
            data = data.to(device)
            
            # Reset hidden states
            model.reset_states()
            
            # Reshape data for time-steps processing
            data = data.permute(1, 0, 2, 3, 4)  # [time_steps, batch_size, channels, height, width]
            
            # Run network for all time steps and collect final time step features
            for t in range(time_steps):
                if isinstance(model, SpikingCNN):
                    # Process through the network
                    x = model.conv1(data[t])
                    spk1, mem1 = model.lif1(x)
                    x = model.pool1(spk1)
                    
                    x = model.conv2(x)
                    spk2, mem2 = model.lif2(x)
                    x = model.pool2(spk2)
                    
                    x = model.conv3(x)
                    spk3, mem3 = model.lif3(x)
                    x = model.pool3(spk3)
                    
                    x = x.view(x.size(0), -1)
                    x = model.fc(x)
                    x = model.dropout(x)
                    spk4, mem4 = model.lif4(x)
                    
                    # Save final layer features
                    if t == time_steps - 1:
                        features = spk4.cpu().numpy()
                else:  # ResNet
                    # Forward pass using the model's forward method
                    # We'll use a hook to capture features
                    _, _ = model(data[t])
            
            # Append features and labels
            all_features.append(features)
            all_labels.append(targets.cpu().numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)

def plot_tsne(features, labels, output_path):
    """Create t-SNE visualization of features"""
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_tsne[:, 0], 
        features_tsne[:, 1], 
        c=labels, 
        cmap='tab10', 
        alpha=0.7,
        s=5
    )
    
    plt.colorbar(scatter, label='Digit Class')
    plt.title('t-SNE Visualization of SNN Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function for SNN activation visualization"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(args, device)
    model.eval()
    
    # Prepare dataset
    test_loader = prepare_dataset(args)
    
    # Dictionary to store activations
    activations = {}
    
    # Register hooks
    hooks = register_hooks(model, activations, args.visualize_all)
    
    # Get one batch for visualization
    data_iter = iter(test_loader)
    data, targets = next(data_iter)
    data = data.to(device)
    
    # Create directory for spike and membrane visualizations
    spike_dir = os.path.join(args.output_dir, 'spikes')
    membrane_dir = os.path.join(args.output_dir, 'membrane')
    os.makedirs(spike_dir, exist_ok=True)
    os.makedirs(membrane_dir, exist_ok=True)
    
    print("Visualizing activations for one batch...")
    
    # Process each sample in time steps
    with torch.no_grad():
        # Reset hidden states
        model.reset_states()
        
        # Reshape data for time-steps processing
        data = data.permute(1, 0, 2, 3, 4)  # [time_steps, batch_size, channels, height, width]
        
        # Initialize containers for spike recordings
        spike_recordings = {layer: [] for layer in activations.keys()}
        
        # Process each time step
        for t in range(args.time_steps):
            # Forward pass
            spk_out, mem_out = model(data[t])
            
            # Store activations for each layer
            for layer_name, layer_output in activations.items():
                # Layer output is a tuple (spikes, membrane)
                spike_recordings[layer_name].append(layer_output)
        
        # For each layer, stack time steps and visualize
        for layer_name, recordings in spike_recordings.items():
            # Stack recordings over time
            spk_rec = torch.stack([rec[0] for rec in recordings])  # (time, batch, ...)
            mem_rec = torch.stack([rec[1] for rec in recordings])  # (time, batch, ...)
            
            # Visualize spike pattern
            visualize_spike_pattern(
                spk_rec, 
                f'Spike Pattern - {layer_name}', 
                os.path.join(spike_dir, f'spike_pattern_{layer_name}.png')
            )
            
            # Visualize membrane potential
            visualize_membrane_potential(
                mem_rec, 
                f'Membrane Potential - {layer_name}', 
                os.path.join(membrane_dir, f'membrane_potential_{layer_name}.png')
            )
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create t-SNE visualization if requested
    if args.visualize_all:
        print("Collecting features for t-SNE visualization...")
        
        # Dictionary to store features for each layer
        features_dict = {}
        
        # Register hooks for the hidden layers we want to visualize
        if isinstance(model, SpikingCNN):
            layers_to_visualize = ['lif1', 'lif2', 'lif3', 'lif4', 'lif_out']
        else:  # ResNet
            layers_to_visualize = ['lif1', 'res1_lif2', 'res2_lif2', 'fc_lif', 'out_lif']
        
        # Create t-SNE directory
        tsne_dir = os.path.join(args.output_dir, 'tsne')
        os.makedirs(tsne_dir, exist_ok=True)
        
        # Collect features and create t-SNE plot for the final fully connected layer
        features, labels = collect_features_for_tsne(model, test_loader, device, args.time_steps)
        plot_tsne(features, labels, os.path.join(tsne_dir, 'tsne_features.png'))
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
