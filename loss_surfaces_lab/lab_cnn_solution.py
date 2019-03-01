import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data
from util import load_mnist

def get_mnist():
    def get_loader(X, y, batch_size=64):
        train = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), 
                                           torch.from_numpy(y))
        train_loader = torch.utils.data.DataLoader(train, 
                                                   batch_size=batch_size)
        return train_loader
    # shuffle data
    X_train, y_train, X_test, y_test = load_mnist()
    np.random.seed(0)
    idxs = np.random.permutation(np.arange(X_train.shape[0]))
    X_train, y_train = X_train[idxs], y_train[idxs]
    train_loader = get_loader(X_train, y_train) 
    test_loader = get_loader(X_test, y_test)
    return train_loader, test_loader

class CNN(nn.Module):
    def __init__(self, k=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6*k, kernel_size=5)
        self.conv2 = nn.Conv2d(6*k, 16*k, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
def train_epoch(model, optimizer, train_loader, criterion, device):
    """
    for each batch 
    performs forward and backward pass and parameters update 
    
    Input:
    model: instance of model (example defined above)
    optimizer: instance of optimizer (defined above)
    train_loader: instance of DataLoader
    
    Returns:
    nothing
    
    Do not forget to set net to train mode!
    """
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).long()

        optimizer.zero_grad()
        output = model(x_batch)
        
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    

def evaluate_loss_acc(loader, model, criterion, device):
    """
    Evaluates loss and accuracy on the whole dataset
    
    Input:
    loader:  instance of DataLoader
    model: instance of model (examle defined above)
    
    Returns:
    (loss, accuracy)
    
    Do not forget to set net to eval mode!
    """
    with torch.no_grad():
        cumloss, cumacc = 0, 0
        num_objects = 0
        model.eval()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()

            output = model(x_batch)
            loss = criterion(output, y_batch)

            pred = torch.max(output, 1)[1]
            acc = torch.sum(pred == y_batch) 

            cumloss += loss.item()
            cumacc += acc.item()
            num_objects += len(x_batch)
    return cumloss / num_objects, cumacc / num_objects
    
    
def train(model, opt, train_loader, test_loader, criterion, n_epochs, device, verbose=True, path_to_save=""):
    """
    Performs training of the model and prints progress
    
    Input:
    model: instance of model (example defined above)
    opt: instance of optimizer 
    train_loader: instance of DataLoader
    test_loader: instance of DataLoader (for evaluation)
    n_epochs: int
    
    Returns:
    4 lists: train_log, train_acc_log, val_log, val_acc_log
    with corresponding metrics per epoch
    """
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    for epoch in range(n_epochs):
        train_epoch(model, opt, train_loader, criterion, device)
        train_loss, train_acc = evaluate_loss_acc(train_loader, model, criterion, device)
        val_loss, val_acc = evaluate_loss_acc(test_loader, model, criterion, device)

        train_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_log.append(val_loss)
        val_acc_log.append(val_acc)
        
        if verbose:
             print (('Epoch [%d/%d], Loss (train/test): %.4f/%.4f,'+\
               ' Acc (train/test): %.4f/%.4f' )
                   %(epoch+1, n_epochs, \
                     train_loss, val_loss, train_acc, val_acc))
        if path_to_save is not None:
            torch.save(model.state_dict(), \
                       path_to_save+"/model_ep%d.cpt"%epoch)
            
    return train_log, train_acc_log, val_log, val_acc_log

