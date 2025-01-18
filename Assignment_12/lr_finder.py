import torch
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
from transformer import GPT, GPTConfig
from train import DataLoaderLite
import json
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def run_lr_finder(start_lr, end_lr, num_iter=100, device='cuda', writer=None):
    """
    Run a single iteration of the learning rate finder
    """
    # Initialize model, data and optimizer
    batches, no_of_tokens = 16, 256
    train_loader = DataLoaderLite(B=batches, T=no_of_tokens)
    model = GPT(GPTConfig())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=0.01)
    
    print(f"Running LR finder from {start_lr:.2e} to {end_lr:.2e}")
    
    # Initialize the LR finder
    lr_finder = LRFinder(model, optimizer, criterion=None, device=device)
    
    # Run the range test and get the learning rates and losses
    lr_finder.range_test(
        train_loader,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode="exp",
        val_loader=None,
        suggest_lr=True
    )
    
    # Log to tensorboard if writer is provided
    if writer is not None:
        # Get the learning rates and losses from the finder
        lrs = lr_finder.history['lr']
        losses = lr_finder.history['loss']
        
        # Log each point to tensorboard
        for i, (lr, loss) in enumerate(zip(lrs, losses)):
            writer.add_scalar('LR_Finder/loss', loss, i)
            writer.add_scalar('LR_Finder/learning_rate', lr, i)
    
    # Plot and save the results
    fig = plt.figure(figsize=(10, 6))
    lr_finder.plot()
    plt.title(f'LR Range Test ({start_lr:.2e} to {end_lr:.2e})')
    plt.savefig(f'lr_finder_range_{start_lr:.2e}_{end_lr:.2e}.png')
    plt.close()
    
    suggested_lr = lr_finder.suggestion()
    return suggested_lr

def iterative_lr_search(iterations=3, initial_range=(1e-7, 10)):
    """
    Perform multiple iterations of LR finding, narrowing the range each time
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join("logs", "lr_finder", datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    results = []
    current_start, current_end = initial_range
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Create a new figure for this iteration in tensorboard
        with writer.as_default():
            writer.add_scalar('LR_Finder/iteration', i+1, 0)
        
        # Run LR finder with tensorboard writer
        suggested_lr = run_lr_finder(current_start, current_end, device=device, writer=writer)
        results.append({
            'iteration': i+1,
            'start_lr': current_start,
            'end_lr': current_end,
            'suggested_lr': suggested_lr
        })
        
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        
        # Log the suggested learning rate for this iteration
        writer.add_scalar('LR_Finder/suggested_lr', suggested_lr, i)
        
        # Narrow the range around the suggested LR for next iteration
        current_start = suggested_lr / 10
        current_end = suggested_lr * 10
        
        # Save results after each iteration
        with open('lr_finder_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    # Print final summary
    print("\nLR Finder Summary:")
    print("-" * 50)
    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Range: {result['start_lr']:.2e} - {result['end_lr']:.2e}")
        print(f"  Suggested LR: {result['suggested_lr']:.2e}")
    
    writer.close()
    return results[-1]['suggested_lr']  # Return the final suggested LR

if __name__ == "__main__":
    # Run the iterative LR search
    final_lr = iterative_lr_search(iterations=3)
    print(f"\nFinal suggested learning rate: {final_lr:.2e}")
    print("\nYou can now use this learning rate in your training script.")
