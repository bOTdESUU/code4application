# PhD Application Code Repository

This repository contains three independent sub-repositories showcasing different aspects of my programming and research skills:
## Note
The EMG-Decomp-FastICA repository is fully documented, while LFQ-BeT and Latent-Plan repositories contain experimental code developed during master's thesis research with minimal documentation due to project scope and time constraints.

## 1. EMG-Decomp-FastICA
Implementation of a fast ICA-based sEMG signal decomposition framework using PyTorch with GPU acceleration.
- Well-documented codebase
- Developed during research internship
- Focuses on efficient signal processing using modern deep learning tools

## 2. LFQ-BeT
Look-up Free Quantization implementation for transformer-based language models in robotics manipulation.
- Built on PyTorch Lightning and nanoGPT templates
- Adapted from VQ-BeT repository with custom modifications
- Key features:
    - Custom GPT model implementation
    - Pretrained autoencoder integration
    <!-- - LFQ implementation with large codebook support -->
    - Modified BeT structure without offset head to experiment with LFQ with large codebook
    - Custom dataset handling and robotic simulation pipeline

## 3. Latent-Plan
Experimental implementation of Look-up Free Quantization within the Trajectory Autoencoder Planner (TAP) framework in robotic application.
- Modified TAP model architecture
- Replaced Vector Quantization with LFQ
- Performance evaluation using existing TAP pipeline


## Research Context
This work explores the effectiveness of Look-up Free quantization discrete representation as latent representation for transformer-based language models in robotics manipulation applications.