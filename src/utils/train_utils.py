from src.models import SpikingCNN, SpikingResNet
import torch.nn as nn
import torch.nn.functional as F
import torch


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