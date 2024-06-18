import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scripts.data.dataset import load_dataset
from scripts.metric import accuracy_score
from scripts.metric.acc import binary_accuracy
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def train(epoch,
          model: nn.Module, 
          criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, 
          dataloader: DataLoader, 
          scheduler = None, 
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          save_path="models/saved",
          is_train = True,
          **kwargs):
    
    epoch_loss = 0
    epoch_acc = 0
    
    for data, label, lengths, dfs in tqdm(dataloader, total=len(dataloader), desc="Train Loop", position=1, leave=False, ncols=200):
        data = data.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        logit = model(data)
        loss = criterion(logit, label)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        prob = F.softmax(logit, dim=1)
        acc = binary_accuracy(prob.argmax(dim=1), label)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    
    print(f'\n{epoch+1} Epoch Loss: {epoch_loss/len(dataloader)}, Epoch ACC: {epoch_acc/len(dataloader)}')
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, f"1000data_{epoch+1}_epoch.pth"))
        print(f"\t{epoch+1} Model Saved")
    
    return epoch_loss, epoch_acc
