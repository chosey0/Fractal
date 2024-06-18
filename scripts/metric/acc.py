import torch
def binary_accuracy(pred, label):
    # A simple function to calculate accuracy
    rounded_preds = torch.round(pred)
    correct = (rounded_preds == label).float()
    acc = correct.sum() / len(correct)
    return acc