"""
Training utility functions for SNN models
Contains helper functions for training, testing, and model setup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Import SNNTorch
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikegen
import snntorch.spikeplot as splt

# Import from src modules
from src.models import SpikingCNN, SpikingResNet

def get_loss_function(args):
    """Get the appropriate loss function based on arguments"""
    if args.loss == 'ce':
        return nn.CrossEntropyLoss()
    elif args.loss == 'mse':
        return nn.MSELoss()
    elif args.loss == 'count':
        # Custom spike count loss that rewards higher spike counts for correct class
        def spike_count_loss(outputs, targets):
            # One-hot encode targets
            targets_one_hot = F.one_hot(targets, num_classes=10).float()
            # Calculate MSE between spike counts and targets
            return F.mse_loss(outputs, targets_one_hot)
        return spike_count_loss
    else:
        return nn.CrossEntropyLoss()

def get_optimizer(model, args):
    """Get the appropriate optimizer based on arguments"""
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def get_scheduler(optimizer, args):
    """Get the learning rate scheduler if requested"""
    if args.scheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        return None

def get_model(args, device):
    """Get the appropriate SNN model based on arguments"""
    if args.model == 'cnn':
        return SpikingCNN(
            input_channels=2,  # N-MNIST has 2 channels (positive and negative events)
            num_time_bins=args.time_steps,
            beta=args.beta,
            threshold=args.threshold,
            dropout_p=args.dropout
        ).to(device)
    elif args.model == 'resnet':
        return SpikingResNet(
            input_channels=2,
            num_time_bins=args.time_steps,
            beta=args.beta,
            threshold=args.threshold,
            alpha=args.alpha,
            dropout_p=args.dropout
        ).to(device)
    else:
        return SpikingCNN(
            input_channels=2,
            num_time_bins=args.time_steps,
            beta=args.beta,
            threshold=args.threshold,
            dropout_p=args.dropout
        ).to(device)

def forward_pass(model, data, time_steps):
    """Perform a forward pass through the SNN model for given time steps"""
    # Reshape data for time-steps processing [time_steps, batch_size, channels, height, width]
    data = data.permute(1, 0, 2, 3, 4)
    
    # Initialize lists for output spikes
    spk_rec = []
    
    # Reset model states
    model.reset_states()
    
    # Run network for multiple timesteps
    for t in range(time_steps):
        spk_out = model(data[t])
        spk_rec.append(spk_out)
    
    # Stack output spikes - shape: [time_steps, batch_size, num_classes]
    spk_rec = torch.stack(spk_rec)
    
    # Sum over time dimension to get rate coding
    outputs = spk_rec.sum(dim=0)
    
    return outputs

def train(model, train_loader, optimizer, criterion, epoch, args, device, printer):
    """Training function for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    # Create progress bar
    pbar = printer.create_progress_bar(
        train_loader, 
        desc=f'Epoch {epoch+1}/{args.epochs}', 
        unit='batch'
    )
    
    for i, (data, targets) in enumerate(pbar):
        data = data.to(device)  # [batch_size, time_steps, channels, height, width]
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = forward_pass(model, data, args.time_steps)
        
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
        batch_loss = total_loss / (i + 1)
        batch_acc = 100. * correct / total
        printer.update_progress_bar(pbar, batch_loss, batch_acc)
    
    train_time = time.time() - start_time
    return total_loss/len(train_loader), 100.*correct/total, train_time

def test(model, test_loader, criterion, args, device, printer):
    """Evaluation function on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # Create progress bar for test set
    test_pbar = printer.create_progress_bar(test_loader, desc='Testing', unit='batch')
    
    with torch.no_grad():
        for data, targets in test_pbar:
            data = data.to(device)  # [batch_size, time_steps, channels, height, width]
            targets = targets.to(device)
            
            # Forward pass
            outputs = forward_pass(model, data, args.time_steps)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Store predictions and targets for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            batch_loss = total_loss / (len(all_predictions) / args.batch_size)
            batch_acc = 100. * correct / total
            printer.update_progress_bar(test_pbar, batch_loss, batch_acc)
    
    return total_loss/len(test_loader), 100.*correct/total, np.array(all_predictions), np.array(all_targets)