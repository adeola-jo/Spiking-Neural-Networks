"""
Main training script for N-MNIST dataset using SNNTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse
from models import SpikingCNN
from utils import save_checkpoint, load_checkpoint, accuracy, plot_results, save_results

def parse_args():
    parser = argparse.ArgumentParser(description='N-MNIST training with SNNTorch')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--time_steps', type=int, default=25, help='Number of time steps for simulation')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for LIF neurons')
    parser.add_argument('--threshold', type=float, default=1.0, help='Firing threshold for LIF neurons')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Directory for datasets')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--test_only', action='store_true', help='Only run inference on test set')
    return parser.parse_args()

def prepare_dataset(args):
    """Prepare the N-MNIST dataset"""
    # Define transformations
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=(2, 34, 34), time_window=args.time_steps)
    ])
    
    # Load the datasets
    train_dataset = tonic.datasets.NMNIST(root=args.data_dir, train=True, transform=frame_transform, download=True)
    test_dataset = tonic.datasets.NMNIST(root=args.data_dir, train=False, transform=frame_transform, download=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def train(model, train_loader, optimizer, criterion, epoch, args, device):
    """Training function for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch')
    for i, (data, targets) in enumerate(pbar):
        data = data.to(device)  # [batch_size, time_steps, channels, height, width]
        targets = targets.to(device)
        
        # Initialize hidden states and outputs
        utils.reset(model)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Reshape data for time-steps processing
        # [time_steps, batch_size, channels, height, width]
        data = data.permute(1, 0, 2, 3, 4)
        
        # Initialize spike record for all time steps
        spk_rec = []
        mem_rec = []
        
        for t in range(args.time_steps):
            # Pass data per time step
            spk_out, mem_out = model(data[t])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        
        # Stack spike recordings to get [time_steps, batch_size, output_features]
        spk_rec = torch.stack(spk_rec)
        
        # Sum output spikes and divide by number of time steps for rate coding
        spike_count = torch.sum(spk_rec, dim=0)
        outputs = spike_count / args.time_steps  # Rate coding
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': total_loss/(i+1), 'acc': 100.*correct/total})
    
    return total_loss/len(train_loader), 100.*correct/total

def test(model, test_loader, criterion, args, device):
    """Evaluation function on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Testing', unit='batch'):
            data = data.to(device)
            targets = targets.to(device)
            
            # Initialize hidden states and outputs
            utils.reset(model)
            
            # Reshape data for time-steps processing
            # [time_steps, batch_size, channels, height, width]
            data = data.permute(1, 0, 2, 3, 4)
            
            # Initialize spike record for all time steps
            spk_rec = []
            
            for t in range(args.time_steps):
                # Pass data per time step
                spk_out, _ = model(data[t])
                spk_rec.append(spk_out)
            
            # Stack spike recordings
            spk_rec = torch.stack(spk_rec)
            
            # Sum output spikes and divide by number of time steps for rate coding
            spike_count = torch.sum(spk_rec, dim=0)
            outputs = spike_count / args.time_steps  # Rate coding
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return total_loss/len(test_loader), 100.*correct/total

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare datasets and dataloaders
    train_loader, test_loader = prepare_dataset(args)
    
    # Initialize model
    model = SpikingCNN(
        input_channels=2,  # N-MNIST has 2 channels (positive and negative events)
        num_time_bins=args.time_steps,
        beta=args.beta,
        threshold=args.threshold
    ).to(device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tracking variables
    start_epoch = 0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0
    
    # Resume training if specified
    if args.resume:
        checkpoint = load_checkpoint(args.save_dir, model, optimizer)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            train_accs = checkpoint['train_accs']
            test_accs = checkpoint['test_accs']
            print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
    
    # Test only mode
    if args.test_only:
        test_loss, test_acc = test(model, test_loader, criterion, args, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        return
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args, device)
        
        # Evaluate on test set
        test_loss, test_acc = test(model, test_loader, criterion, args, device)
        
        # Print results
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Save tracking variables
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
        }, is_best, args.save_dir)
    
    # Plot results
    plot_results(train_losses, test_losses, train_accs, test_accs, args.save_dir)
    
    # Save final results
    save_results({
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc,
    }, args.save_dir)
    
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
