"""
Training script for N-MNIST dataset using SNNTorch
Implements a complete training pipeline with support for different SNN models
"""
import sys
import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from src modules
from src.models import SpikingCNN, SpikingResNet
from src.utils import save_checkpoint, load_checkpoint, plot_results, save_results, visualize_nmnist_sample
from src.data import prepare_nmnist_dataset
from src.utils.print_utils import get_printer  # Import our new printing utilities

# Import our training utilities
from train_utils import get_model, get_loss_function, get_optimizer, get_scheduler, train, test

def parse_args():
    parser = argparse.ArgumentParser(description='N-MNIST training with SNNTorch')
    # Dataset parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--time_steps', type=int, default=25, help='Number of time steps for simulation')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Directory for datasets')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'], help='Model architecture')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for LIF neurons')
    parser.add_argument('--threshold', type=float, default=1.0, help='Firing threshold for LIF neurons')
    parser.add_argument('--alpha', type=float, default=0.9, help='Alpha parameter for adaptive neurons')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse', 'count'], 
                        help='Loss function: cross-entropy (ce), mean squared error (mse), or spike count (count)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], 
                        help='Optimizer: adam, adamw, or sgd')
    parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
    
    # Utility parameters
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for saving logs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--test_only', action='store_true', help='Only run inference on test set')
    parser.add_argument('--visualize', action='store_true', help='Visualize training results')
    
    # Printing parameters
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no_color', action='store_true', help='Disable colored output')
    return parser.parse_args()

def run_test_only(model, test_loader, criterion, args, device, printer):
    """Run only testing on the model"""
    printer.header("Test Only Mode")
    test_loss, test_acc, predictions, targets = test(model, test_loader, criterion, args, device, printer)
    printer.success(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Plot confusion matrix
    if args.visualize:
        printer.info("Generating classification report and confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(args.log_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Print classification report
        report = classification_report(targets, predictions)
        print("\nClassification Report:")
        print(report)
        
        # Save classification report
        with open(os.path.join(args.log_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        printer.success(f"Saved visualizations to {args.log_dir}")

def visualize_epoch_results(model, test_loader, epoch, args, printer):
    """Create visualizations for the current epoch"""
    printer.info("Generating visualizations...")
    
    # Get a random test batch for visualization
    data_iter = iter(test_loader)
    data, targets = next(data_iter)
    
    # Select a random sample
    idx = np.random.randint(0, data.shape[0])
    sample = data[idx]
    target = targets[idx]
    
    # Visualize the sample
    fig = visualize_nmnist_sample(sample, target.item())
    plt.savefig(os.path.join(args.log_dir, f'sample_epoch_{epoch+1}.png'))
    plt.close(fig)
    printer.info(f"Saved sample visualization for epoch {epoch+1}")
    
    # If using the ResNet model, also save some weight visualizations
    if args.model == 'resnet':
        # Visualize first layer filters
        conv1_weights = model.conv1.weight.data.cpu().numpy()
        plt.figure(figsize=(12, 6))
        for i in range(min(16, conv1_weights.shape[0])):
            plt.subplot(4, 4, i+1)
            plt.imshow(conv1_weights[i, 0], cmap='viridis')
            plt.axis('off')
        plt.suptitle('First Layer Conv Filters')
        plt.savefig(os.path.join(args.log_dir, f'filters_epoch_{epoch+1}.png'))
        plt.close()
        printer.info("Saved filter visualizations")

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Initialize our fancy printer
    printer = get_printer(
        verbose=args.verbose, 
        log_file=os.path.join(args.log_dir, 'training.log'),
        use_color=not args.no_color
    )
    
    printer.header("SNN Training for N-MNIST Dataset")
    printer.info(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Print configuration
    printer.print_config(vars(args))
    
    # Prepare datasets and dataloaders
    printer.info("Preparing datasets...")
    train_loader, test_loader = prepare_nmnist_dataset(args.data_dir, args.batch_size, args.time_steps)
    
    # Initialize model
    printer.info("Initializing model...")
    model = get_model(args, device)
    printer.print_model_summary(model)
    
    # Initialize optimizer, loss, and scheduler
    printer.info("Setting up training components...")
    optimizer = get_optimizer(model, args)
    criterion = get_loss_function(args)
    scheduler = get_scheduler(optimizer, args)
    
    # Initialize tracking variables
    start_epoch = 0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    train_times = []
    best_acc = 0
    
    # Resume training if specified
    if args.resume:
        printer.info("Attempting to resume from checkpoint...")
        checkpoint = load_checkpoint(args.save_dir, model, optimizer)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            train_accs = checkpoint['train_accs']
            test_accs = checkpoint['test_accs']
            train_times = checkpoint.get('train_times', [])
            printer.success(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
        else:
            printer.warning("No checkpoint found, starting from scratch.")
    
    # Test only mode
    if args.test_only:
        run_test_only(model, test_loader, criterion, args, device, printer)
        return
    
    # Training loop
    printer.header("Starting Training")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # Print epoch header
        printer.epoch_header(epoch + 1, args.epochs)
        
        # Train for one epoch
        train_loss, train_acc, train_time = train(model, train_loader, optimizer, criterion, epoch, args, device, printer)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print training progress
        printer.training_progress(train_time, train_loss, train_acc, current_lr)
        
        # Update learning rate scheduler if used
        if scheduler:
            scheduler.step()
        
        # Evaluate on test set
        test_loss, test_acc, predictions, targets = test(model, test_loader, criterion, args, device, printer)
        
        # Print testing progress
        printer.testing_progress(test_loss, test_acc)
        
        # Save tracking variables
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_times.append(train_time)
        
        # Check if this is the best model
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        
        # Print epoch summary
        printer.epoch_summary(
            epoch + 1, train_loss, train_acc, 
            test_loss, test_acc, train_time,
            best_acc, is_best
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"{args.model}_checkpoint.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'train_times': train_times,
        }, is_best, args.save_dir)
        
        # Print checkpoint saved message
        printer.checkpoint_saved(checkpoint_path, is_best)
        
        # Create and save visualizations periodically
        if args.visualize and (epoch % 5 == 0 or epoch == args.epochs - 1):
            visualize_epoch_results(model, test_loader, epoch, args, printer)
    
    # Calculate total training time
    total_train_time = time.time() - training_start_time
    
    # Print final results
    printer.training_complete(total_train_time, best_acc)
    
    # Plot final results
    if args.visualize:
        printer.info("Creating final visualization plots...")
        plot_results(train_losses, test_losses, train_accs, test_accs, args.log_dir)
        
        # Plot training time per epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_times) + 1), train_times, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time per Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(args.log_dir, 'training_time.png'))
        plt.close()
    
    # Print results table
    printer.print_results_table({
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_times': train_times
    })
    
    # Save final results
    save_results({
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_times': train_times,
        'best_acc': best_acc,
        'args': vars(args),
    }, args.log_dir)
    
    printer.success(f"All results saved to {args.log_dir}")

if __name__ == "__main__":
    main()