"""
Printing utilities for SNN training scripts
Provides colorful, formatted output with icons and tables
"""
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from tabulate import tabulate
from termcolor import colored
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define color constants
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
# Define emoji/icon constants
class Icons:
    INFO = "ℹ️ "
    SUCCESS = "✅ "
    WARNING = "⚠️ "
    ERROR = "❌ "
    TRAINING = "🔄 "
    TESTING = "🔍 "
    TIME = "⏱️ "
    CHECKPOINT = "💾 "
    CONFIG = "⚙️ "
    EPOCHS = "📊 "
    BRAIN = "🧠 "
    FIRE = "🔥 "
    NEURON = "⚡ "
    CHART = "📈 "
    
class SNNPrinter:
    def __init__(self, verbose: bool = True, log_file: Optional[str] = None, use_color: bool = True):
        """
        Initialize the printer with verbose setting and optional log file
        
        Args:
            verbose: Whether to print detailed information
            log_file: Path to log file for output
            use_color: Whether to use colored output (disable for some environments)
        """
        self.verbose = verbose
        self.log_file = log_file
        self.use_color = use_color
        self.start_time = time.time()
        
        # Create log file directory if it doesn't exist
        if log_file:
            # Get the directory part (if any)
            log_dir = os.path.dirname(log_file)
            
            # Only create directories if there's a directory path specified
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Clear the log file
            with open(log_file, 'w') as f:
                f.write(f"=== SNN Training Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def _log(self, message: str):
        """Log message to file if log_file is specified"""
        if self.log_file:
            with open(self.log_file, 'a') as f:
                # Strip ANSI color codes for log file
                clean_message = message
                for color_code in vars(Colors).values():
                    if isinstance(color_code, str) and color_code.startswith('\033'):
                        clean_message = clean_message.replace(color_code, '')
                f.write(clean_message + '\n')
    
    def _colorize(self, text: str, color: str, bold: bool = False) -> str:
        """Add color to text if use_color is enabled"""
        if not self.use_color:
            return text
        bold_code = Colors.BOLD if bold else ''
        return f"{bold_code}{color}{text}{Colors.ENDC}"
    
    def header(self, message: str):
        """Print a header message"""
        formatted = self._colorize(f"\n{Icons.INFO} {message}", Colors.HEADER, True)
        print(formatted)
        self._log(formatted)
        
        # Add a separator line
        separator = self._colorize("=" * (len(message) + 4), Colors.HEADER)
        print(separator)
        self._log(separator)
    
    def info(self, message: str):
        """Print an informational message"""
        formatted = self._colorize(f"{Icons.INFO} {message}", Colors.CYAN)
        print(formatted)
        self._log(formatted)
    
    def success(self, message: str):
        """Print a success message"""
        formatted = self._colorize(f"{Icons.SUCCESS} {message}", Colors.GREEN)
        print(formatted)
        self._log(formatted)
    
    def warning(self, message: str):
        """Print a warning message"""
        formatted = self._colorize(f"{Icons.WARNING} {message}", Colors.YELLOW)
        print(formatted)
        self._log(formatted)
    
    def error(self, message: str):
        """Print an error message"""
        formatted = self._colorize(f"{Icons.ERROR} {message}", Colors.RED, True)
        print(formatted)
        self._log(formatted)
    
    def print_config(self, config: Dict[str, Any]):
        """Print configuration parameters in a table"""
        if not self.verbose:
            return
            
        self.header(f"{Icons.CONFIG} Configuration")
        
        # Group parameters by category
        categories = {
            "Dataset": ["batch_size", "time_steps", "data_dir"],
            "Model": ["model", "beta", "threshold", "alpha", "dropout"],
            "Training": ["epochs", "lr", "weight_decay", "loss", "optimizer", "scheduler"],
            "Utility": ["save_dir", "log_dir", "resume", "device", "test_only", "visualize"]
        }
        
        # Print each category
        for category, params in categories.items():
            table_data = []
            for param in params:
                if param in config:
                    value = config[param]
                    # Format boolean values
                    if isinstance(value, bool):
                        value = "✓" if value else "✗"
                    table_data.append([param, value])
            
            print(self._colorize(f"\n{category}:", Colors.BOLD))
            print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="pretty"))
        
        print()  # Add a blank line after config
        self._log(f"Configuration: {config}")
    
    def print_model_summary(self, model, input_shape=None):
        """Print model summary with layer information"""
        if not self.verbose:
            return
            
        self.header(f"{Icons.BRAIN} Model Summary: {model.__class__.__name__}")
        
        # Count parameters - only if model has parameters() method (PyTorch models)
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(self._colorize(f"Total parameters: {total_params:,}", Colors.CYAN))
            print(self._colorize(f"Trainable parameters: {trainable_params:,}", Colors.CYAN))
            
            param_info = f", Parameters: {trainable_params:,}"
        except (AttributeError, TypeError):
            # For non-PyTorch models or our dummy test model
            print(self._colorize("Parameter count not available for this model type", Colors.YELLOW))
            param_info = ""
        
        # Print model structure
        print(self._colorize("\nModel Structure:", Colors.BOLD))
        print(model)
        print()  # Add a blank line after model summary
        
        self._log(f"Model Summary: {model.__class__.__name__}{param_info}")
    
    def epoch_header(self, epoch: int, total_epochs: int):
        """Print header for beginning of epoch"""
        percentage = int((epoch / total_epochs) * 100)
        progress_bar = self._create_progress_bar(percentage, 20)
        message = f"{Icons.EPOCHS} Epoch {epoch}/{total_epochs} {progress_bar} {percentage}%"
        
        formatted = self._colorize(f"\n{message}", Colors.BLUE, True)
        print(formatted)
        self._log(formatted)
    
    def _create_progress_bar(self, percentage: int, length: int = 20) -> str:
        """Create a text-based progress bar"""
        filled_length = int(length * percentage / 100)
        bar = '█' * filled_length + '░' * (length - filled_length)
        return f"[{bar}]"
    
    def training_progress(self, epoch_time: float, loss: float, accuracy: float, lr: Optional[float] = None):
        """Print training progress for current epoch"""
        data = [
            ["Time", f"{epoch_time:.2f}s", Icons.TIME],
            ["Loss", f"{loss:.4f}", Icons.FIRE],
            ["Accuracy", f"{accuracy:.2f}%", Icons.SUCCESS]
        ]
        
        if lr is not None:
            data.append(["Learning Rate", f"{lr:.6f}", Icons.CONFIG])
        
        table = tabulate(data, headers=["Metric", "Value", ""], tablefmt="simple")
        print(self._colorize(f"{Icons.TRAINING} Training:", Colors.GREEN))
        print(table)
        self._log(f"Training - Time: {epoch_time:.2f}s, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    def testing_progress(self, test_loss: float, test_accuracy: float):
        """Print testing progress for current epoch"""
        data = [
            ["Loss", f"{test_loss:.4f}", Icons.FIRE],
            ["Accuracy", f"{test_accuracy:.2f}%", Icons.SUCCESS]
        ]
        
        table = tabulate(data, headers=["Metric", "Value", ""], tablefmt="simple")
        print(self._colorize(f"{Icons.TESTING} Testing:", Colors.BLUE))
        print(table)
        self._log(f"Testing - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    def epoch_summary(self, epoch: int, train_loss: float, train_acc: float, 
                     test_loss: float, test_acc: float, time_taken: float,
                     best_acc: float, is_best: bool = False):
        """Print comprehensive summary at the end of an epoch"""
        if not self.verbose:
            # Print minimal summary
            status = Icons.SUCCESS + " (Best)" if is_best else ""
            print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}% {status}")
            return
            
        data = [
            ["", "Loss", "Accuracy"],
            ["Training", f"{train_loss:.4f}", f"{train_acc:.2f}%"],
            ["Testing", f"{test_loss:.4f}", f"{test_acc:.2f}%"]
        ]
        
        print(self._colorize(f"\n{Icons.CHART} Epoch {epoch} Summary:", Colors.BOLD))
        print(tabulate(data, headers="firstrow", tablefmt="pretty"))
        
        print(self._colorize(f"Time: {time_taken:.2f}s", Colors.CYAN))
        
        if is_best:
            best_msg = f"{Icons.SUCCESS} New best accuracy: {test_acc:.2f}% (previous: {best_acc:.2f}%)"
            print(self._colorize(best_msg, Colors.GREEN, True))
        
        # Elapsed time since start
        elapsed = time.time() - self.start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_msg = f"{Icons.TIME} Total elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        print(self._colorize(time_msg, Colors.CYAN))
        
        self._log(f"Epoch {epoch} Summary - Train: {train_loss:.4f}/{train_acc:.2f}%, Test: {test_loss:.4f}/{test_acc:.2f}%, Best: {best_acc:.2f}%")
    
    def checkpoint_saved(self, path: str, is_best: bool = False):
        """Print message when checkpoint is saved"""
        status = "best model" if is_best else "checkpoint"
        message = f"{Icons.CHECKPOINT} Saved {status} to {path}"
        
        formatted = self._colorize(message, Colors.GREEN)
        print(formatted)
        self._log(formatted)
    
    def training_complete(self, train_time: float, best_acc: float):
        """Print training completion message with summary"""
        hours, rem = divmod(train_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        self.header(f"{Icons.SUCCESS} Training Complete")
        
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        print(self._colorize(f"{Icons.TIME} Total training time: {time_str}", Colors.CYAN))
        print(self._colorize(f"{Icons.CHART} Best accuracy: {best_acc:.2f}%", Colors.GREEN, True))
        
        self._log(f"Training Complete - Total time: {time_str}, Best accuracy: {best_acc:.2f}%")
    
    def print_results_table(self, epoch_results: Dict[str, List[float]]):
        """Print a summary table of results across all epochs"""
        if not self.verbose:
            return
            
        self.header(f"{Icons.CHART} Results Summary")
        
        # Create table data
        headers = ["Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc", "Time (s)"]
        table_data = []
        
        for i in range(len(epoch_results['train_losses'])):
            row = [
                i+1,
                f"{epoch_results['train_losses'][i]:.4f}",
                f"{epoch_results['train_accs'][i]:.2f}%",
                f"{epoch_results['test_losses'][i]:.4f}",
                f"{epoch_results['test_accs'][i]:.2f}%",
                f"{epoch_results['train_times'][i]:.2f}"
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        
        # Print best epoch
        best_idx = np.argmax(epoch_results['test_accs'])
        best_epoch = best_idx + 1
        best_acc = epoch_results['test_accs'][best_idx]
        
        best_msg = f"{Icons.SUCCESS} Best performance at epoch {best_epoch} with accuracy {best_acc:.2f}%"
        print(self._colorize(best_msg, Colors.GREEN, True))
        
        self._log(f"Results Summary - Best epoch: {best_epoch}, Best accuracy: {best_acc:.2f}%")

    def update_progress_bar(self, pbar, loss, accuracy):
        """Update tqdm progress bar with current metrics"""
        pbar.set_postfix({
            'loss': f"{loss:.4f}", 
            'acc': f"{accuracy:.2f}%"
        })

    def create_progress_bar(self, iterable, desc, unit='batch'):
        """Create a tqdm progress bar"""
        return tqdm(iterable, desc=desc, unit=unit, 
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                    disable=not self.verbose)

# Function to get a printer instance
def get_printer(verbose=True, log_file=None, use_color=True):
    """Factory function to get a printer instance"""
    return SNNPrinter(verbose=verbose, log_file=log_file, use_color=use_color)

# Test function
if __name__ == "__main__":
    # Test the printer
    print("Testing the printer...")
    printer = get_printer(verbose=True, log_file="test_log.txt")
    
    printer.header("SNN Training")
    
    # Test config printing
    test_config = {
        "batch_size": 128,
        "time_steps": 25,
        "data_dir": "./datasets",
        "model": "resnet",
        "beta": 0.5,
        "threshold": 1.0,
        "alpha": 0.9,
        "dropout": 0.2,
        "epochs": 10,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "loss": "ce",
        "optimizer": "adam",
        "scheduler": True,
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
        "resume": False,
        "device": "cuda",
        "test_only": False,
        "visualize": True
    }
    printer.print_config(test_config)
    
    # Test model summary
    class DummyModel:
        def __init__(self):
            pass
        def __str__(self):
            return "DummyModel(layers=[Conv2d, LIF, MaxPool2d, Linear, LIF])"
    
    # Test with the basic dummy model (no parameters method)
    printer.print_model_summary(DummyModel())
    
    # Test with a mock PyTorch-like model
    try:
        import torch.nn as nn
        
        class MockTorchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
                self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
                self.fc = nn.Linear(32 * 8 * 8, 10)
                
            def forward(self, x):
                return self.fc(self.conv2(self.conv1(x)).view(-1, 32 * 8 * 8))
            
            def __str__(self):
                return "MockTorchModel(Conv2d->Conv2d->Linear)"
        
        print("\nTesting with PyTorch-like model:")
        printer.print_model_summary(MockTorchModel())
    except ImportError:
        print("\nSkipping PyTorch model test (torch not available)")
    
    # Test epoch progress
    printer.epoch_header(1, 10)
    printer.training_progress(45.23, 0.7823, 68.45, 0.001)
    printer.testing_progress(0.6543, 72.34)
    printer.epoch_summary(1, 0.7823, 68.45, 0.6543, 72.34, 45.23, 72.34, True)
    
    # Test checkpoint saved
    printer.checkpoint_saved("./checkpoints/model_epoch1.pth", True)
    
    # Test training complete
    printer.training_complete(3600, 85.67)
    
    # Test results table
    test_results = {
        "train_losses": [0.9823, 0.7654, 0.5432],
        "train_accs": [62.34, 72.45, 79.87],
        "test_losses": [0.8765, 0.6543, 0.4321],
        "test_accs": [65.43, 75.67, 82.34],
        "train_times": [45.23, 43.67, 44.12]
    }
    printer.print_results_table(test_results)
    
    print("Testing complete!")