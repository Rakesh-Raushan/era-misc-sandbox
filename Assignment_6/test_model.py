import torch
import pytest
from network import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def has_batch_norm(model):
    return any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())

def has_dropout(model):
    return any(isinstance(m, torch.nn.Dropout) for m in model.modules())

def has_gap_or_fc(model):
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    return has_gap or has_fc

def test_parameter_count():
    model = Net()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, which exceeds the limit of 20,000"

def test_batch_normalization():
    model = Net()
    assert has_batch_norm(model), "Model does not use Batch Normalization"

def test_dropout():
    model = Net()
    assert has_dropout(model), "Model does not use Dropout"

def test_gap_or_fc():
    model = Net()
    assert has_gap_or_fc(model), "Model does not use either Global Average Pooling or Fully Connected layer" 