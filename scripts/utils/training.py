import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scripts.data.dataset import load_dataset

def main():
    model = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)