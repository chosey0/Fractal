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

if __name__ == "__main__":
    input_size = 4
    output_size = 2

    model = CNN1D(input_size, output_size)
    model.load_state_dict(torch.load("models/saved/1000data_10_epoch.pth"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    df = read_data("data/test/20240618_선진뷰티사이언스.csv", read_function=pd.read_csv, callback=calc_fractal, n=500)
    # fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
    # fplt.plot(df["fractal_low"], style="o", color='#00ffbc', width=2.0)
    # fplt.plot(df["fractal_high"], style="o", color='#ffbcff', width=2.0)
    # fplt.show()
    df["model_fractal_low"] = np.nan
    df["model_fractal_high"] = np.nan
    
    with torch.no_grad():
        model.eval()
        test(df, model, device=device)
