from torch.utils.data import Dataset

class FractalDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # self.dataset[dataset["label"] == 1] = dataset[dataset["label"] == 1].iloc[:int(len(dataset[dataset["label"] == 0])*1.5)]
        # self.dataset = self.dataset.dropna()
        print(self.dataset["shape"].max(), self.dataset["shape"].min())
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset.iloc[idx]