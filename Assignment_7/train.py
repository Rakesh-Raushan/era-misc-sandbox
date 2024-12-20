import torch
import torch.optim as optim
from torchvision import datasets, transforms
from models.model import Model_1, Model_2, Model_3
from utils.utils import train, test
import random
import numpy as np
from torchsummary import summary

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # Training settings
    batch_size = 128
    epochs = 15
    lr = 0.01
    seed = 1

    # Check for CUDA or MPS availability
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Select device
    if use_cuda:
        print(f"Using CUDA")
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif use_mps:
        print(f"Using MPS")
        device = torch.device("mps")
        torch.mps.manual_seed(seed)
        # torch.mps.manual_seed_all(seed)
        # torch.backends.mps.deterministic = True
        # torch.backends.mps.benchmark = False
    else:
        print("Using CPU")
        device = torch.device("cpu")
        
    # DataLoader kwargs
    train_kwargs = {
        'batch_size': batch_size,
        'worker_init_fn': seed_worker,
        'generator': torch.Generator().manual_seed(seed),
        'shuffle': True
    }
    test_kwargs = {
        'batch_size': batch_size,
        'worker_init_fn': seed_worker,
        'generator': torch.Generator().manual_seed(seed),
        'shuffle': False
    }
    
    if use_cuda or use_mps:
        cuda_kwargs = {
            'num_workers': 8,
            'pin_memory': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2, scale=(0.1, 0.25), ratio=(0.3, 3.3), value=0, inplace=False)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_data = datasets.MNIST('../data', train=True, download=True,
                              transform=train_transform)
    test_data = datasets.MNIST('../data', train=False,
                             transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    
    # Initialize model
    model = Model_3()
    print(f"training model3 with total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"\nmodel summary: {summary(model, (1, 28, 28), device='cpu')}")

    # Send model to device
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-9)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Learning rate scheduler
    # try step scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=0
    )
    
    # Training loop
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        print(f'EPOCH: {epoch}')
        train(model, device, train_loader, optimizer)
        test_loss, accuracy = test(model, device, test_loader)
        
        # Step scheduler
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']

        # try step scheduler
        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]
        
        print(f"Test_acc: {accuracy:.1f} Learning Rate: {current_lr:.6f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "mnist_model_best.pth")
    
    print(f'Best Accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main() 