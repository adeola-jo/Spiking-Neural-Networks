"""
Checkpoint utilities for saving and loading model states
"""
import os
import torch
import shutil

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    """
    Save checkpoint to disk
    
    Args:
        state: Dict containing model state, optimizer state, etc.
        is_best: Boolean indicating if this is the best model so far
        save_dir: Directory to save the checkpoint
        filename: Filename for the checkpoint
    """
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_filepath)
        print("Saved best model checkpoint to", best_filepath)

def load_checkpoint(save_dir, model, optimizer=None, filename='checkpoint.pth.tar'):
    """
    Load checkpoint from disk
    
    Args:
        save_dir: Directory containing the checkpoint
        model: Model to load the weights into
        optimizer: Optimizer to load the state into
        filename: Filename of the checkpoint
        
    Returns:
        Dict containing the loaded checkpoint info if successful, None otherwise
    """
    filepath = os.path.join(save_dir, filename)
    if not os.path.isfile(filepath):
        print(f"No checkpoint found at {filepath}")
        return None
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint
