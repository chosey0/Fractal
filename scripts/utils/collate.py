
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd

def collate(batch):
    results = []
    labels = []
    lengths = []
    dfs = []
    
    for buffer, label, shape, length in batch:
        data = torch.frombuffer(buffer, dtype=torch.int64).reshape(shape)[:, 1:]
        use_data = ((torch.roll(data, shifts=-1, dims=0) - data) / data * 100)[:-1, :]
        # use_data = np.frombuffer(buffer, dtype=np.int64).reshape(shape)[:-1, 1:]
        # norm_data = normalize(use_data, norm="max", axis=0)
        # results.append(torch.tensor(norm_data, dtype=torch.float32)) # (현재값 - 이전값) / 이전값 * 100 -> roll에 의해 처음 값이 마지막 값으로 이동했으므로 마지막 값은 제외
        results.append(use_data)
        labels.append(label)
        lengths.append(use_data.shape[0])
        
        # 확인용
        df = pd.DataFrame(data=np.frombuffer(buffer, dtype=np.int64).reshape(shape), columns=["Time", "Open", "High", "Low", "Close", "5", "20", "120"])
        df.index = df["Time"]
        dfs.append(df)


    padded_tensor = pad_sequence(results, batch_first=True, padding_value=100)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32)
    
    return padded_tensor, labels_tensor.long(), lengths, dfs