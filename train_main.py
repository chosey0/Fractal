import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from scripts.data.dataset import load_dataset
from scripts.models.train import train
from scripts.utils.collate import collate

from interface.models import CNN1D
import os
import numpy as np
from tqdm import tqdm
if __name__ == "__main__":
    input_size = 4
    output_size = 2

    model = CNN1D(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = None #optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_set = load_dataset("data/process/kis_minmax1000.pkl")
    temp = train_set.dataset
    weights = np.where(temp["label"] == 0, len(temp[temp["label"] ==0]) / len(temp), len(temp[temp["label"] ==1]) / len(temp))
    
    print("데이터 분포 및 길이:")
    print(f"\tTotal: {len(temp)}")
    for cls in temp["label"].unique():
        print(f"\t{cls}: {len(temp[temp['label'] == cls]) / len(temp)}, {len(temp[temp['label'] == cls])}개")
    
    train_loader = DataLoader(train_set, collate_fn=collate, batch_size=128, sampler = WeightedRandomSampler(weights, len(weights)))
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    n_epochs = 30
    model.train()
    
    total_loss = 0
    total_acc = 0
    
    for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Train Epoch", position=0, ncols=200):
        epoch_loss, epoch_acc = train(epoch, model, criterion, optimizer, train_loader, scheduler=scheduler, save_path="models/saved", device=device)
        total_loss += epoch_loss
        total_acc += epoch_acc
        
        print(f'\nTotal Loss: {total_loss/(epoch+1)}, Total ACC: {total_acc/(epoch+1)}')
    
    # model.eval()