import os
from scripts.data.dataset import read_data, create_dataset, load_dataset
from scripts.data.fractal_indicator import calc_fractal
import pandas as pd
from tqdm import tqdm

import finplot as fplt
import datetime
from pyqtgraph.exporters import SVGExporter
fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = datetime.timezone.utc

def main():
    source = "data/raw/한국투자/day"
    path_list = os.listdir(source)
    
    results = []
    shapes = []
    lengths = []
    labels = []
    
    
    for path in tqdm(path_list, total=len(path_list)):
        if "ma" in path: continue
        # if path == "20240618_선진뷰티사이언스.csv":
        df = read_data(
                os.path.abspath(os.path.join(source, path)), 
                pd.read_csv, 
                calc_fractal,
                window_type="day",
                n=20,
            )
        if df is None: continue    
        # NOTE: calc_fractal 결과 확인용
        # ax0, ax1 = fplt.create_plot(path, rows=2)
        # fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=ax0)
        # fplt.plot(df["fractal_high"], style="o", color='#ffbcff', width=2.0, ax=ax0)
        # fplt.plot(df["fractal_low"], style="o", color='#00ffbc', width=2.0, ax=ax0)
        # fplt.plot(df["5"], color='#973131', width=2.0, ax=ax0)
        # fplt.plot(df["20"], color='#5A639C', width=2.0, ax=ax0)
        # fplt.plot(df["120"], color='#A0937D', width=2.0, ax=ax0)
        # fplt.volume_ocv(df[['Open', 'Close', 'Amount']], ax=ax1)
        # 가공 데이터 이미지 저장용
        # def save():
        #     fplt.screenshot(open(f'img/days/{path.split(".")[0]}.png', 'wb'))
        # fplt.timer_callback(save, .2, single_shot=True)
        # fplt.timer_callback(fplt.close, .4, single_shot=True)
        # fplt.show()
        
        result, label, shape, length = create_dataset(df, use_cols=["Time", "Open", "High", "Low", "Close", "5", "20", "120"], max_len=20)
        results.extend(result)
        labels.extend(label)
        shapes.extend(shape)
        lengths.extend(length)
        
        
    result_df = pd.DataFrame({"data": results, "label": labels, "shape": shapes, "length":lengths})
    result_df.to_pickle("data/process/kis_minmax_day_20.pkl")
    

if __name__ == "__main__":
    main()