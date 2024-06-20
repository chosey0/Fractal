import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scripts.models.test import test

from interface.models import CNN1D
import pandas as pd
import numpy as np
from scripts.data.dataset import read_data
from scripts.data.fractal_indicator import calc_fractal

import finplot as fplt
import datetime

fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = datetime.timezone.utc

def main(model_name):
    input_size = 6
    output_size = 3

    model = CNN1D(input_size, output_size)
    model.load_state_dict(torch.load(f"models/saved/{model_name}.pth"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    df = read_data("data/test/모나리자.csv", read_function=pd.read_csv, callback=calc_fractal, window_type="day", n=20)
    
    df.loc[~(df['20'] > df['120']), ['fractal_high', 'fractal_low']] = np.nan
    ax0, ax1 = fplt.create_plot("모나리자", rows=2)
    fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=ax0)
    fplt.plot(df["fractal_high"], style="o", color='#ffbcff', width=2.0, ax=ax0)
    fplt.plot(df["fractal_low"], style="o", color='#00ffbc', width=2.0, ax=ax0)
    fplt.plot(df["20"], color='#5A639C', width=2.0, ax=ax0)
    fplt.plot(df["120"], color='#A0937D', width=2.0, ax=ax0)
    fplt.volume_ocv(df[['Open', 'Close', 'Amount']], ax=ax1)
    fplt.show()
    
    df["model_fractal_low"] = np.nan
    df["model_fractal_high"] = np.nan
    
    prob_history = {
        "low": [0, 0],
        "high": [0, 0],
        "None": [0, 0]
    }
    
    with torch.no_grad():
        model.eval()
        test(df, model, device=device, max_len=20, prob_history=prob_history)

if __name__ == "__main__":
    main("kis_day20_ma20120_cls3_5_epoch")
    