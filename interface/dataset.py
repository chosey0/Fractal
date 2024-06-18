from torch.utils.data import Dataset

class FractalDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        print(dataset["shape"].max(), dataset["shape"].min())
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset.iloc[idx]