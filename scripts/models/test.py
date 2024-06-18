import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from tqdm import tqdm
import finplot as fplt
import datetime

fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = datetime.timezone.utc

def test(df: pd.DataFrame,
          model: nn.Module, 
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    for i in tqdm(range(len(df)), total=len(df), desc="Test Loop", position=1, leave=False, ncols=200):
        data = df[["Open", "High", "Low", "Close"]].iloc[:i]
        data = torch.tensor(data.values, device=device)
        
        input_data = ((torch.roll(data, shifts=-1, dims=0) - data) / data * 100)[:-1, :].unsqueeze(0).to(torch.float32)
        
        # 모델 제한으로 2개 이상의 데이터 필요
        if input_data.size()[1] < 2:
            continue
        if input_data.size()[1] > 1000:
            input_data = input_data[:, -1000:, :]
        
        logit = model(input_data.to("cuda"))
        prob = F.sigmoid(logit).flatten()#F.sigmoid(out).flatten()

        if prob[0] >= .998:
            df.loc[df.index[i], "model_fractal_low"] = df["Low"].iloc[i]
        elif prob[1] >= .998:
            df.loc[df.index[i], "model_fractal_high"] = df["High"].iloc[i]
        else:
            continue
    
    fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
    fplt.plot(df["model_fractal_low"], style="o", color='#00ffbc', width=2.0)
    fplt.plot(df["model_fractal_high"], style="o", color='#ffbcff', width=2.0)
    fplt.show()
