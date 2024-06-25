import os
from scripts.data.dataset import read_data, create_dataset, load_dataset
from scripts.data.fractal_indicator import calc_fractal
import pandas as pd
from tqdm import tqdm

from scripts.utils.view import view_data
import json

import numpy as np
def main(dataset_name, use_cols):
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
        # scaler = StandardScaler()
        # view_data(path, df, save_img=False, other_data=[(df["fractal_high"], "o", "#ffbcff"), (df["fractal_low"], "o", "#00ffbc"), (df["20"], None, "#5A639C"), (df["120"], None, "#A0937D")])
        # scaled_df = pd.DataFrame(scaler.fit_transform(df[['Open', 'Close', 'High', 'Low', "5", "20", "120"]]), index=df.index, columns=['Open', 'Close', 'High', 'Low', "5", "20", "120"])
        # view_data(path+"_scaled", scaled_df, save_img=False, other_data=[(scaled_df["20"], None, "#5A639C"), (scaled_df["120"], None, "#A0937D")])

        result, label, shape, length = create_dataset(df, use_cols=use_cols, max_len=20)
        results.extend(result)
        labels.extend(label)
        shapes.extend(shape)
        lengths.extend(length)
        
        
    result_df = pd.DataFrame({"data": results, "label": labels, "shape": shapes, "length":lengths})
    dataset_info = {
        "name": dataset_name,
        "source_dir": source,
        "source_size": len(path_list),
        "dataset_size": len(result_df),
        "use_cols": use_cols,
        "max_len": 20,
        "cls_ratio": [f"\t{cls}: {len(result_df[result_df['label'] == cls]) / len(result_df)}, {len(result_df[result_df['label'] == cls])}개" for cls in result_df["label"].unique()]
    }
    
    result_df.to_pickle(f"data/process/{dataset_name}.pkl")
    with open(f"data/process/{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main("kis_day20_ma20120_cls3", ["Time", "Open", "High", "Low", "Close", "20", "120"])