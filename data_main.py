import os
from scripts.data.dataset import read_data, create_dataset, load_dataset
from scripts.data.fractal_indicator import calc_fractal
import pandas as pd
from tqdm import tqdm

import finplot as fplt
import datetime

fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = datetime.timezone.utc

def main():
    source = "data/test"
    path_list = os.listdir(source)
    
    results = []
    shapes = []
    lengths = []
    labels = []
    
    for path in tqdm(path_list, total=len(path_list)):
        if path == "20240618_선진뷰티사이언스.csv":
            df = read_data(
                os.path.abspath(os.path.join(source, path)), 
                pd.read_csv, 
                calc_fractal,
                n=500
                )
            
            # NOTE: calc_fractal 결과 확인용
            # fplt.candle_bull_color = "#ffbbc0"
            # fplt.candle_bull_body_color = "#ffbbc0" 
            # fplt.candle_bear_color = "#bbc0ff"
            # fplt.candle_bear_body_color = "#bbc0ff"
            
            # candle = fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
            # f_high = fplt.plot(df["fractal_high"], style="o", color='#ffbcff', width=2.0)
            # f_low = fplt.plot(df["fractal_low"], style="o", color='#00ffbc', width=2.0)
            # fplt.show()
        else:
            continue
        
        result, label, shape, length = create_dataset(df, 1000)
        results.extend(result)
        labels.extend(label)
        shapes.extend(shape)
        lengths.extend(length)
        
    # result_df = pd.DataFrame({"data": results, "label": labels, "shape": shapes, "length":lengths})
    # result_df.to_pickle("data/process/kis_minmax1000.pkl")

if __name__ == "__main__":
    main()