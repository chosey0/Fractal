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
          model_name = "Test",
          is_train = True,
          **kwargs):
    
    epoch_loss = 0
    epoch_acc = 0
    
    prob_history = {
        "low": [],
        "high": []
    }
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
        
        prob = F.softmax(logit, dim=1) #F.sigmoid(logit)
        acc = binary_accuracy(prob.argmax(dim=1), label)
        
        prob_history["low"].extend(prob[:, 0].cpu().tolist())
        prob_history["high"].extend(prob[:, 1].cpu().tolist())
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    
    print(f'\n{epoch+1} Epoch Loss: {epoch_loss/len(dataloader)}, Epoch ACC: {epoch_acc/len(dataloader)}')
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_{epoch+1}_epoch.pth"))
        print(f"\t{epoch+1} Model Saved")
    
    return epoch_loss, epoch_acc, prob_history
