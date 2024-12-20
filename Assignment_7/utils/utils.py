import torch
import torch.nn.functional as F
from tqdm import tqdm

# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     pbar = tqdm(train_loader)
#     for batch_idx, (data, target) in enumerate(pbar):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         pbar.set_description(f'Epoch {epoch} Loss: {loss.item():.6f}')

# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
    
#     print(f'\nTest set: Average loss: {test_loss:.4f}, '
#           f'Accuracy: {correct}/{len(test_loader.dataset)} '
#           f'({accuracy:.2f}%)\n')
    
#     return accuracy 


# define metrics to calc

train_losses = []
train_acc = []

test_losses = []
test_acc = []


def train(model, device, train_loader, optimizer):
    # set model to train mode
    model.train()

    # tqdm iterator
    pbar = tqdm(train_loader)

    # correct and processed vars
    correct = 0
    processed = 0

    # loop on batches of data
    for batch_idx, (data,target) in enumerate(pbar):
        #send data, targte to training device
        data, target = data.to(device), target.to(device)

        # Initialize grad to zero for the fresh batch grad accumuation
        optimizer.zero_grad()

        # pred with model
        y_pred = model(data)

        # calc loss
        batch_loss = F.nll_loss(y_pred, target)
        train_losses.append(batch_loss)

        # backprop loss to calc and acc grad w.r.t loss of batch
        batch_loss.backward()

        # update weights as per losses seen in this batch
        optimizer.step()

        # calculate correct pred count and acc for batch
        pred_labels = y_pred.argmax(dim=1, keepdim=True)
        correct_count_batch = pred_labels.eq(target.view_as(pred_labels)).sum().item()

        # update total correct and total processed so far
        correct+= correct_count_batch
        processed+= len(data)

        # set pbar desc
        pbar.set_description(desc=f'batch Loss = {batch_loss.item()} batch_id = {batch_idx} accuracy = {100*correct/processed:.02f}'
                            )
        #append train acc
        train_acc.append(100*correct/processed)


def test(model, device, test_loader):
    # set model to eval mode
    model.eval()

    # define var to calc correct and processed
    correct = 0
    processed = 0
    test_loss = 0 # seeing loss as the code runs has no value for test

    # set a no grad context
    with torch.no_grad():
        for data,target in test_loader:
            #send data, target to device
            data, target = data.to(device), target.to(device)

            # do pred
            y_pred = model(data)

            #calc loss for batch as summed and update total test loss
            batch_loss = F.nll_loss(y_pred, target, reduction='sum').item()
            test_loss+= batch_loss
            # collect loss
            test_losses.append(batch_loss)

            # count correct
            pred_labels = y_pred.argmax(dim=1, keepdim=True)
            correct_batch = pred_labels.eq(target.view_as(pred_labels)).sum().item()

            #update correct
            correct+= correct_batch
            processed+= len(data)

    # avg loss on test makes more sense to avg it
    test_loss/= processed
    # collect avg losses
    test_losses.append(test_loss)

    print(f'\n Test set avg loss: {test_loss:.4f} \
                Accuracy: {correct}/{processed}, {100*correct/processed:.2f}'
         )

    test_acc.append(100*correct/processed)

    return test_loss, round(100*correct/processed, 1)

