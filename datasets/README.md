# Spiking Neural Network Datasets

This repository contains information about various datasets suitable for Spiking Neural Network (SNN) projects and research.

## Table of Contents
- [Introduction](#introduction)
- [Traditional Vision Datasets](#traditional-vision-datasets)
- [Neuromorphic/Event-Based Datasets](#neuromorphicevent-based-datasets)
- [Temporal/Time Series Datasets](#temporaltime-series-datasets)
- [Neuromorphic Benchmarks](#neuromorphic-benchmarks)
- [Getting Started](#getting-started)
- [Contributing](#contributing)

## Introduction

Spiking Neural Networks (SNNs) represent a biologically inspired approach to neural network computing. Unlike traditional artificial neural networks, SNNs incorporate the concept of time and communicate through discrete events called "spikes" rather than continuous values.

This README catalogs various datasets suitable for SNN research and development, categorized by type. Each entry includes a brief description, key features, and information on how to access the dataset.

## Traditional Vision Datasets

These datasets were originally designed for conventional neural networks but can be converted to spike-based representations using various encoding techniques.

### MNIST
- **Description**: Handwritten digit recognition (0-9)
- **Size**: 60,000 training images, 10,000 test images
- **Resolution**: 28×28 grayscale
- **Classes**: 10 (digits 0-9)
- **Best for**: Beginners, baseline comparisons
- **Access**: Available through most ML libraries (TorchVision, TensorFlow Datasets)

### Fashion-MNIST
- **Description**: Clothing item classification dataset designed as a more challenging drop-in replacement for MNIST
- **Size**: 60,000 training images, 10,000 test images
- **Resolution**: 28×28 grayscale
- **Classes**: 10 (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot)
- **Best for**: Slightly more challenging baseline than MNIST
- **Access**: Available through most ML libraries (TorchVision, TensorFlow Datasets)

### CIFAR-10/100
- **Description**: Natural image classification dataset
- **Size**: 50,000 training images, 10,000 test images
- **Resolution**: 32×32 color
- **Classes**: 10 (CIFAR-10) or 100 (CIFAR-100)
- **Best for**: More challenging vision tasks, comparing with CNNs
- **Access**: Available through most ML libraries (TorchVision, TensorFlow Datasets)

## Neuromorphic/Event-Based Datasets

These datasets contain native spike-based data recorded with event cameras or similar neuromorphic sensors.

### N-MNIST
- **Description**: Neuromorphic version of MNIST recorded with event-based cameras
- **Size**: 60,000 training samples, 10,000 test samples
- **Resolution**: 34×34 event streams
- **Classes**: 10 (digits 0-9)
- **Best for**: First steps with native neuromorphic data
- **Access**: https://www.garrickorchard.com/datasets/n-mnist

### DVS128 Gesture
- **Description**: Hand gestures recorded with DVS cameras
- **Size**: 1,342 recordings
- **Resolution**: 128×128 event streams
- **Classes**: 11 (hand gestures)
- **Best for**: Action/motion recognition, temporal pattern learning
- **Access**: https://research.ibm.com/interactive/dvsgesture/

### PokerDVS
- **Description**: Neuromorphic recordings of poker card pips (symbols)
- **Size**: 131 recordings
- **Resolution**: 128×128 event streams
- **Classes**: 4 (clubs, diamonds, hearts, spades)
- **Duration**: ~35ms per sample
- **Events**: ~5,000 events per sample
- **Best for**: Simple shape recognition, beginner neuromorphic projects
- **Access**: Through Tonic library or original repository

### N-Caltech101
- **Description**: Event-based version of the Caltech101 dataset
- **Size**: 8,709 samples
- **Resolution**: 240×180 event streams
- **Classes**: 101 object categories
- **Best for**: Advanced object recognition with event data
- **Access**: https://www.garrickorchard.com/datasets/n-caltech101

### DAVIS Driving Dataset
- **Description**: Automotive scenes recorded with DAVIS event cameras
- **Features**: Includes both frame and event data
- **Duration**: Multiple hours of driving scenarios
- **Best for**: Autonomous driving applications, object tracking
- **Access**: https://docs.prophesee.ai/stable/datasets.html

### IBM DVS Falls
- **Description**: Human fall detection recordings for healthcare applications
- **Size**: 29 subjects, various scenarios
- **Best for**: Healthcare monitoring, anomaly detection
- **Access**: https://www.research.ibm.com/dvsgesture/

### ASL-DVS
- **Description**: American Sign Language gestures recorded with event cameras
- **Size**: 24 gestures, multiple subjects
- **Classes**: 24 (letters from American Sign Language)
- **Best for**: Human-computer interaction, gesture recognition
- **Access**: Available through neuromorphic dataset repositories

## Temporal/Time Series Datasets

These datasets focus on temporal patterns and can be effectively processed by SNNs.

### UCI HAR
- **Description**: Human Activity Recognition dataset from smartphone sensor data
- **Size**: 10,299 sequences
- **Features**: Accelerometer and gyroscope readings
- **Classes**: 6 activities (walking, walking upstairs, walking downstairs, sitting, standing, laying)
- **Best for**: Human activity recognition, mobile computing applications
- **Access**: UCI Machine Learning Repository

### Speech Commands
- **Description**: Spoken word classification dataset
- **Size**: 65,000 one-second audio clips
- **Classes**: 35 common words
- **Best for**: Audio processing, speech recognition
- **Access**: TensorFlow Datasets

### SHD (Spiking Heidelberg Digits)
- **Description**: Spoken digit dataset pre-processed as spike trains
- **Size**: 10,000 samples
- **Features**: 700 input neurons × ~250 time steps
- **Classes**: 10 (spoken digits)
- **Best for**: Temporal pattern recognition with spikes
- **Access**: https://compneuro.net/datasets/

### SSC (Spiking Speech Command)
- **Description**: Speech commands pre-processed into spike trains
- **Features**: Various encoding methods available
- **Classes**: 10 or 35 commands
- **Best for**: Benchmarking speech recognition with SNNs
- **Access**: Available through SNN frameworks

## Neuromorphic Benchmarks

These are specialized datasets and benchmarks designed specifically for evaluating SNN performance.

### Heidelberg Spiking Datasets
- **Description**: Collection of datasets specifically for SNN benchmarking
- **Includes**: SHD, SSC, and other specialized datasets
- **Best for**: Standardized performance evaluation
- **Access**: https://compneuro.net/datasets/

### SnnTorch Tutorials Datasets
- **Description**: Example datasets included with the SnnTorch library
- **Features**: Pre-processed for direct use with SnnTorch
- **Best for**: Learning SnnTorch framework
- **Access**: Included with SnnTorch installation

## Getting Started

For beginners to SNNs, we recommend starting with:
1. **MNIST** or **N-MNIST** - for basic classification tasks
2. **PokerDVS** - for simple neuromorphic data processing
3. **SHD** - for temporal pattern recognition

These datasets have extensive documentation and example implementations available online.

## Contributing

Contributions to this list are welcome! If you know of a dataset that should be included, please submit a pull request with the relevant information.

## License

This README is provided under the MIT License. The datasets themselves have their own specific licenses, which should be checked before use.