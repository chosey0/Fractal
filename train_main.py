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
import pickle
from collections import Counter

def main(dataset_name):
    input_size = 6
    output_size = 3

    model = CNN1D(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = None #optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_set = load_dataset(f"data/process/{dataset_name}.pkl")
    
    # 클래스 불균형 조정
    temp = train_set.dataset
    class_weights = {label: len(temp) / count for label, count in Counter(temp["label"]).items()}
    sample_weights = [class_weights[label] for label in temp["label"].values]
    norm_weights = np.array(sample_weights) / np.sum(sample_weights)
    
    print("데이터 분포 및 길이:")
    print(f"\tTotal: {len(temp)}")
    
    for cls in temp["label"].unique():
        print(f"\t{cls}: {len(temp[temp['label'] == cls]) / len(temp)}, {len(temp[temp['label'] == cls])}개")
    
    train_loader = DataLoader(train_set, collate_fn=collate, batch_size=128, sampler = WeightedRandomSampler(norm_weights, len(norm_weights)))
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    n_epochs = 10
    
    total_loss = 0
    total_acc = 0
    
    history = {
        "info":{
            "name": dataset_name,
            "architecture": "CNN1D",
            "input_size": input_size,
            "output_size": output_size,
            "n_epochs": n_epochs,
            "optimizer": "Adam",
            "scheduler": "None",

        },
        "history":{
            
        }
    }
    
    model.train()
    for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Train Epoch", position=0, ncols=200):
        epoch_loss, epoch_acc, prob_history = train(epoch, model, criterion, optimizer, train_loader, scheduler=scheduler, save_path="models/saved", model_name = dataset_name, device=device)

        history["history"][epoch] = {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "prob_history": prob_history
        }
    
    # import finplot as fplt
    # for probs in history["prob_history"]:
    #     ax1, ax2 = fplt.create_plot("Test", rows=2)
    #     fplt.plot(probs["low"], ax=ax1)
    #     fplt.plot(probs["high"], ax=ax2)
    #     fplt.show()
        
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    with open(f"models/saved/{dataset_name}.json", "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    

    
    
if __name__ == "__main__":
    main("kis_day20_ma20120_cls3")
    # model.eval()