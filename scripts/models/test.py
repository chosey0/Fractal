import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from tqdm import tqdm
import finplot as fplt
import datetime
from sklearn.preprocessing import minmax_scale, StandardScaler

scaler = StandardScaler()

fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = datetime.timezone.utc

def test(df: pd.DataFrame,
          model: nn.Module, 
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          prob_history = {},
          max_len = 2000):
    color_map = {
        "low": "#973131",
        "high": "#5A639C",
        "None": "#A0937D"
    }
    for i in tqdm(range(len(df)), total=len(df), desc="Test Loop", position=1, leave=False, ncols=200):
        data = df[["Open", "High", "Low", "Close", "20", "120"]].iloc[:i]
        
        if data.empty or len(data) < 2:
            continue
        elif len(data) > max_len:
            data = data.iloc[-max_len:]
            
        input_data = torch.tensor(scaler.fit_transform(data.values), device=device, dtype=torch.float32).to(device)
        
        if len(input_data.size()) == 2:
            input_data = input_data.unsqueeze(0)
        # input_data = ((torch.roll(data, shifts=-1, dims=0) - data) / data * 100)[:-1, :].unsqueeze(0).to(torch.float32)
        
        # # 모델 제한으로 2개 이상의 데이터 필요
        # if input_data.size()[1] > max_len:
        #     input_data = input_data[:, -max_len:, :]
        
        logit = model(input_data)
        prob = F.softmax(logit, dim=1).flatten()# F.sigmoid(logit).flatten()

        for idx, key in enumerate(prob_history):
            prob_history[key].append(prob[idx].item())

        if 1. > prob[0] >= .95:
            df.loc[df.index[i], "model_fractal_low"] = df["Low"].iloc[i]
        elif 1. > prob[1] >= .95:
            df.loc[df.index[i], "model_fractal_high"] = df["High"].iloc[i]
        else:
            continue
    
    ax0, ax1 = fplt.create_plot("Test", rows=2)
    ax1.showGrid(True, True)
    for key in prob_history.keys():
        df[f"{key}_prob"] = prob_history[key]
    
    fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=ax0)
    fplt.plot(df["model_fractal_low"], style="o", color='#00ffbc', width=2.0, ax=ax0)
    fplt.plot(df["model_fractal_high"], style="o", color='#ffbcff', width=2.0, ax=ax0)
    # fplt.plot(df["5"], color='#973131', width=2.0, ax=ax0)
    fplt.plot(df["20"], color='#5A639C', width=2.0, ax=ax0)
    fplt.plot(df["120"], color='#A0937D', width=2.0, ax=ax0)
    
    for key in prob_history.keys():
        df[f"{key}_prob"] = prob_history[key]
        fplt.plot(df[f"{key}_prob"], style="o", color=color_map[key], width=1, ax=ax1)

    fplt.show()
