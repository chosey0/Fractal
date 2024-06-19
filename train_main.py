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

def main(dataset_name):
    input_size = 7
    output_size = 2

    model = CNN1D(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = None #optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_set = load_dataset(f"data/process/{dataset_name}.pkl")
    temp = train_set.dataset
    weights = np.where(temp["label"] == 0, len(temp[temp["label"] == 0]) / len(temp), len(temp[temp["label"] == 1]) / len(temp))
    
    print("데이터 분포 및 길이:")
    print(f"\tTotal: {len(temp)}")
    for cls in temp["label"].unique():
        print(f"\t{cls}: {len(temp[temp['label'] == cls]) / len(temp)}, {len(temp[temp['label'] == cls])}개")
    
    train_loader = DataLoader(train_set, collate_fn=collate, batch_size=128, sampler = WeightedRandomSampler(weights, len(weights)))
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    n_epochs = 10
    
    total_loss = 0
    total_acc = 0
    
    history = {
        "loss": [],
        "acc": [],
        "prob_history": []
    }
    
    model.train()
    for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Train Epoch", position=0, ncols=200):
        epoch_loss, epoch_acc, prob_history = train(epoch, model, criterion, optimizer, train_loader, scheduler=scheduler, save_path="models/saved", model_name = dataset_name, device=device)
        total_loss += epoch_loss
        total_acc += epoch_acc
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        history["prob_history"].append(prob_history)
        print(f'\nTotal Loss: {total_loss/(epoch+1)}, Total ACC: {total_acc/(epoch+1)}')
    
    # import finplot as fplt
    # for probs in history["prob_history"]:
    #     ax1, ax2 = fplt.create_plot("Test", rows=2)
    #     fplt.plot(probs["low"], ax=ax1)
    #     fplt.plot(probs["high"], ax=ax2)
    #     fplt.show()
        
    import pandas as pd
    import matplotlib.pyplot as plt
    
    history_df = pd.DataFrame(history)
    history_df.to_pickle(f"models/saved/{dataset_name}_history.pkl")
    
    # plt.subplot(211)
    # plt.plot(history_df["loss"])
    # plt.title("Loss")
    
    # plt.subplot(212)
    # plt.plot(history_df["acc"])
    # plt.title("ACC")
    # plt.show()
    
    
if __name__ == "__main__":
    main("kis_minmax_day_20")
    # model.eval()