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
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          max_len = 2000):
    
    prob_history = {
        "low": [0, 0,0],
        "high": [0, 0,0]
    }
    for i in tqdm(range(len(df)), total=len(df), desc="Test Loop", position=1, leave=False, ncols=200):
        data = df[["Open", "High", "Low", "Close", "5", "20", "120"]].iloc[:i]
        data = torch.tensor(data.values, device=device)
        
        input_data = ((torch.roll(data, shifts=-1, dims=0) - data) / data * 100)[:-1, :].unsqueeze(0).to(torch.float32)
        
        # 모델 제한으로 2개 이상의 데이터 필요
        if input_data.size()[1] < 2:
            continue
        if input_data.size()[1] > max_len:
            input_data = input_data[:, -max_len:, :]
        
        logit = model(input_data.to("cuda"))
        prob = F.sigmoid(logit).flatten()#F.sigmoid(out).flatten()

        prob_history["low"].append(prob[0].item())
        prob_history["high"].append(prob[1].item())
        if 1. > prob[0] >= .99:
            df.loc[df.index[i], "model_fractal_low"] = df["Low"].iloc[i]
        elif 1. > prob[1] >= .99:
            df.loc[df.index[i], "model_fractal_high"] = df["High"].iloc[i]
        else:
            continue
    
    ax0, ax1, ax2 = fplt.create_plot("Test", rows=3)
    df["model_low"] = prob_history["low"]
    df["model_high"] = prob_history["high"]
    fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=ax0)
    fplt.plot(df["model_fractal_low"], style="o", color='#00ffbc', width=2.0, ax=ax0)
    fplt.plot(df["model_fractal_high"], style="o", color='#ffbcff', width=2.0, ax=ax0)
    fplt.plot(df["5"], color='#973131', width=2.0, ax=ax0)
    fplt.plot(df["20"], color='#5A639C', width=2.0, ax=ax0)
    fplt.plot(df["120"], color='#A0937D', width=2.0, ax=ax0)
    fplt.plot(df["model_low"], ax=ax1)
    fplt.plot(df["model_high"], ax=ax2)
    fplt.show()
