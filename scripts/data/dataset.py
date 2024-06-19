import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable

from interface.dataset import FractalDataset

# TODO: 한국투자증권 HTS에서 다운로드 받은 캔들 데이터만 처리 가능 (데이터 상 연도가 3개년도가 있는 경우는 고려안함)
def read_data(path: str, read_function: Callable = pd.read_csv, callback: Callable[[str], None] = print, sep=",", window_type="min", **kwargs):
    
    data = read_function(path, encoding='utf-8', sep=sep, header="infer", engine='python')
    if window_type == "min":
        data["시간"] = pd.to_datetime(str(datetime.datetime.now().year) + "/" + data['시간'], format='%Y/%m/%d,%H:%M')
        try:
            prev_year_point = data[data['시간'].dt.month == 1].index[-1]
            data.loc[prev_year_point+1:, "시간"] = pd.to_datetime(data.loc[prev_year_point+1:, "시간"].dt.strftime(f'{datetime.datetime.now().year - 1}-%m-%d %H%M%S'), format="%Y-%m-%d %H%M%S")
        except IndexError:
            pass
    elif window_type == "day":
        data["시간"] = pd.to_datetime(data["시간"], format="%Y-%m-%d")
    
    # 불필요한 문자 제거
    data.replace(",", "", regex=True, inplace=True)
    data.dropna(inplace=True)
    
    # 데이터 타입 변환
    data[["시가", "고가", "저가", "종가", "거래량", "거래대금"]] = data[["시가", "고가", "저가", "종가", "거래량", "거래대금"]].astype(np.int64)
    data[["5", "20", "120"]] = data[["5", "20", "120"]].astype(np.float64)
    data = data[::-1].reset_index(drop=True) # 최근 -> 과거 순에서 과거 -> 최근 순으로 정렬
    data.rename({"시간": "Time", "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume", "거래대금": "Amount"}, axis=1, inplace=True)
    
    if window_type == "min":
        try:
            data["date_only"] = data["Time"].dt.date
            ma = read_function(path.split(".")[0] + "_ma.csv", encoding='utf-8', sep=sep, header="infer", engine='python')
            ma["시간"] = pd.to_datetime(ma["시간"], format="%Y-%m-%d").dt.date
            ma.replace(",", "", regex=True, inplace=True)
            ma[["5", "20", "120"]] = ma[["5", "20", "120"]].astype(np.float64)
            ma.dropna(inplace=True)
        
            data = data.merge(ma[["시간", "5", "20", "120"]], left_on="date_only", right_on="시간", how="left")
            data.drop(columns=['date_only', '시간'], inplace=True)
        except FileNotFoundError:
            return None
    
    data.set_index(pd.DatetimeIndex(data["Time"]).as_unit("ms").asi8, inplace=True)
    
    return callback(data, **kwargs)

def create_dataset(df: pd.DataFrame, use_cols=["Time", "Open", "High", "Low", "Close", "5", "20", "120"], max_len: int = 2000):
    results = []
    shapes = []
    lengths = []
    labels = []
    
    low_index = df["fractal_low"].dropna().index
    high_index = df["fractal_high"].dropna().index
    # other_index = df[df.index == df.dropna().index].index
    
    for idx in tqdm(low_index, total=len(low_index)):
        label = 0
        chunk_df = df.loc[:idx, use_cols]
        
        if len(chunk_df) > max_len:
            chunk_df = chunk_df[-max_len:]

        chunk_df[use_cols] = chunk_df[use_cols].astype(np.int64)
        results.append(chunk_df[use_cols].values.tobytes()) # 시고저종
        labels.append(label)
        shapes.append(chunk_df[use_cols].values.shape)
        lengths.append(len(chunk_df))
        
    for idx in tqdm(high_index, total=len(high_index)):
        label = 1
        chunk_df = df.loc[:idx, use_cols]
        
        chunk_df[use_cols] = chunk_df[use_cols].astype(np.int64)
        if len(chunk_df) > max_len:
            chunk_df = chunk_df[-max_len:]
            
        results.append(chunk_df[use_cols].values.tobytes()) # 시고저종
        labels.append(label)
        shapes.append(chunk_df[use_cols].values.shape)
        lengths.append(len(chunk_df))
            
    return results, labels, shapes, lengths

def load_dataset(path: str, read_function: Callable = pd.read_pickle, sort_target="shape", **kwargs):
    dataset = read_function(path, **kwargs)
    dataset.sort_values(sort_target, inplace=True)
    return FractalDataset(dataset)

# NOTE: 미완성
def temp_function(df):
    chunk = []
    result = []
    
    for i, row in df.iterrows():
        if np.isnan(row.fractal_high) and np.isnan(row.fractal_low):
            chunk.append(row)

        if not np.isnan(row.fractal_high) or not np.isnan(row.fractal_low):
            chunk.append(row)
            chunk_df = pd.DataFrame(chunk)

            # if chunk_df[~np.isnan(chunk_df["fractal_low"])].empty or chunk_df[~np.isnan(chunk_df["fractal_high"])].empty or len(chunk_df) < 3:
            #     del chunk_df

            #     continue

            if len(chunk_df) < 10:
                chunk = []
                del chunk_df
                continue
            result.append(chunk_df)
            chunk = []

            if result == []:
                if chunk_df.iloc[0].Close < chunk_df.iloc[-1].Close:
                    label = True
                else:
                    label = False
            else:
                if np.isnan(chunk_df.iloc[-1].fractal_high): # 하락
                    label = False
                elif np.isnan(chunk_df.iloc[-1].fractal_low): # 상승
                    label = True