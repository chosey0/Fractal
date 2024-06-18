import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable

from interface.dataset import FractalDataset

# TODO: 한국투자증권 HTS에서 다운로드 받은 1분 캔들 데이터만 처리 가능
def read_data(path: str, read_function: Callable = pd.read_csv, callback: Callable[[str], None] = print, sep=",", **kwargs):
    
    data = read_function(path, encoding='utf-8', sep=sep, header="infer", engine='python')
    data["시간"] = pd.to_datetime(str(datetime.datetime.now().year) + "/" + data['시간'], format='%Y/%m/%d,%H:%M')
    prev_year_point = data[data['시간'].dt.month == 1].index[-1]
    data.loc[prev_year_point+1:, "시간"] = pd.to_datetime(data.loc[prev_year_point+1:, "시간"].dt.strftime(f'{datetime.datetime.now().year - 1}-%m-%d %H%M%S'), format="%Y-%m-%d %H%M%S")
    
    # 불필요한 문자 제거
    data.replace(",", "", regex=True, inplace=True)
    data.dropna(inplace=True)
    
    # 데이터 타입 변환
    data[["시가", "고가", "저가", "종가"]] = data[["시가", "고가", "저가", "종가"]].astype(np.int64)
    data = data[::-1].reset_index(drop=True) # 최근 -> 과거 순에서 과거 -> 최근 순으로 정렬
    data.rename({"시간": "Time", "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close"}, axis=1, inplace=True)

    data.set_index(pd.DatetimeIndex(data["Time"]).as_unit("ms").asi8, inplace=True)
    return callback(data, **kwargs)

def create_dataset(df: pd.DataFrame, max_len: int = 2000):
    results = []
    shapes = []
    lengths = []
    labels = []
    low_index = df["fractal_low"].dropna().index
    high_index = df["fractal_high"].dropna().index
    # other_index = df[df.index == df.dropna().index].index
    
    for idx in tqdm(low_index, total=len(low_index)):
        label = 0
        chunk_df = df.loc[:idx, ["Time", "Open", "High", "Low", "Close"]]
        
        if len(chunk_df) > max_len:
            chunk_df = chunk_df[-max_len:]

        results.append(chunk_df[["Time", "Open", "High", "Low", "Close"]].values.tobytes()) # 시고저종
        labels.append(label)
        shapes.append(chunk_df[["Time", "Open", "High", "Low", "Close"]].values.shape)
        lengths.append(len(chunk_df))
        
    for idx in tqdm(high_index, total=len(high_index)):
        label = 1
        chunk_df = df.loc[:idx, ["Time", "Open", "High", "Low", "Close"]]
        
        if len(chunk_df) > max_len:
            chunk_df = chunk_df[-max_len:]
            
        results.append(chunk_df[["Time", "Open", "High", "Low", "Close"]].values.tobytes()) # 시고저종
        labels.append(label)
        shapes.append(chunk_df[["Time", "Open", "High", "Low", "Close"]].values.shape)
        lengths.append(len(chunk_df))
            
    return results, labels, shapes, lengths

def load_dataset(path: str, read_function: Callable = pd.read_pickle, sort_target="shape", **kwargs):
    dataset = read_function(path, **kwargs)
    dataset.sort_values(sort_target, inplace=True)
    return FractalDataset(dataset)