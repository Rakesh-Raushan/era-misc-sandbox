# MNIST Classification Assignment

This project implements a CNN model to achieve 99.4% test accuracy on the MNIST dataset with the following constraints:
- Less than 8000 parameters
- Less than 15 epochs
- Consistent accuracy (not a one-time achievement)

## Project Structure

```
mnist_project/
├── models/
│   ├── init.py
│   ├── model.py # Contains Model_1, Model_2, Model_3 implementations
├── utils/
│   ├── init.py
│   ├── utils.py # Training and testing utilities
├── train.py # Main training script
├── README.md
├── .gitignore
```

## Models

### Model 1: Baseline
- Target: Initial baseline model to understand the problem
- Parameters: 6,922
- Best Training Accuracy: 98.83%
- Best Test Accuracy: 98.67%
- Analysis: Model shows overfitting and needs regularization

### Model 2: With Regularization
- Target: Add dropout regularization
- Parameters: 6,922
- Best Training Accuracy: 98.22%
- Best Test Accuracy: 98.65%
- Analysis: Reduced overfitting but not reaching target accuracy

### Model 3: Final Model
- Target: Improve model capacity and efficiency
- Parameters: 7,394
- Best Training Accuracy: 97.48%
- Best Test Accuracy: 99.45%
- Analysis: Achieves >99.4% accuracy consistently in final epochs
- Key improvements:
  - Added pooling layer
  - Image augmentation (Rotation and RandomErasing)
  - Optimized learning rate scheduling

## Requirements
- PyTorch
- torchvision
- tqdm
- numpy

## Hardware Support
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS)
- CPU fallback

## Usage

1. Install dependencies:

```bash
pip install torch torchvision tqdm numpy
```

2. Run training:

```bash
python train.py
```

## Training Features
- Data augmentation with random rotation and erasing
- Learning rate scheduling
- Model checkpointing (saves best model)
- Progress bars with tqdm
- Deterministic training with seed setting

## Results
The final model (Model_3) achieves:
- Parameters: 7,394 (under 8k limit)
- Consistent >99.4% accuracy in final epochs
- Trains in 14 epochs (under 15 epoch limit)
- Last 4 epochs: 99.44%, 99.40%, 99.43%, 99.45%

## License
MIT

