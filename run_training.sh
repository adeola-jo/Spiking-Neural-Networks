#!/bin/bash
# Example script to run SNN training with different configurations

# Basic training with verbose output and visualizations
echo "Starting training with CNN model..."
python scripts/train_nmnist.py \
    --model cnn \
    --epochs 20 \
    --batch_size 128 \
    --verbose \
    --visualize

# Training with ResNet model
echo "Starting training with ResNet model..."
python train_nmnist.py \
    --model resnet \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.0005 \
    --scheduler \
    --verbose \
    --visualize

# Quick test run with no color (for logging to file)
echo "Running test-only mode..."
python train_nmnist.py \
    --model cnn \
    --test_only \
    --no_color \
    --verbose

# Minimal output training (no verbose)
echo "Running training with minimal output..."
python train_nmnist.py \
    --model cnn \
    --epochs 10

# Example with custom directories
echo "Running training with custom directories..."
python train_nmnist.py \
    --model resnet \
    --data_dir ./custom_data \
    --save_dir ./custom_checkpoints \
    --log_dir ./custom_logs \
    --verbose