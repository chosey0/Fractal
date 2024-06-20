
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale, StandardScaler

scaler = StandardScaler()
def collate(batch):
    results = []
    labels = []
    lengths = []
    dfs = []
    
    for buffer, label, shape, length in batch:
        raw_data = np.frombuffer(buffer, dtype=np.int64).reshape(shape)
        data = scaler.fit_transform(raw_data)
        use_data = torch.tensor(data, dtype=torch.float32)[:, 1:]
        # data = torch.frombuffer(buffer, dtype=torch.int64).reshape(shape)[:, 1:]
        # use_data = ((torch.roll(data, shifts=-1, dims=0) - data) / data * 100)[:-1, :]
        results.append(use_data)
        labels.append(label)
        lengths.append(use_data.shape[0])
        
        # 확인용
        df = pd.DataFrame(data=np.frombuffer(buffer, dtype=np.int64).reshape(shape))
        df.index = df[0]
        dfs.append(df)


    padded_tensor = pad_sequence(results, batch_first=True, padding_value=100)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32)
    
    return padded_tensor, labels_tensor.long(), lengths, dfs