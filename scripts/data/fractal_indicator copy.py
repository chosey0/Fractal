import numpy as np
import pandas as pd

def calc_fractal(df: pd.DataFrame, n=500):
    # 프랙탈을 계산할 기간 (일반적으로 5일을 사용)

    high_indices = []
    low_indices = []
    def fractal_high(series):
        center = n // 2
        # nonlocal df
        # nonlocal high_indices
        
        # idx = high_indices.pop(0)
        # if "거래대금" in df.columns:
        #     df[""]
        # if np.std(series) == 0:
        #     return np.nan
        # elif np.std(series[:center]) == 0:
        #     return np.nan
        # elif np.std(series[center:]) == 0:
        #     return np.nan
        # 변동성 조건
        # elif any(np.abs(np.diff(series[:center]) / series[1:center]) * 100 >= 25.):
        #     return np.nan
        # 이평선 조건
        if series[center] == series.max():
            # if df.loc[idx, "5"] > df.loc[idx, "20"] > df.loc[idx, "120"] and df.loc[idx, "Amount"] >= 50000000000:
            #     return series[center]
            # else:
            #     return np.nan
            return series[center]
        else:
            return np.nan

    def fractal_low(series):
        center = n // 2
        
        # nonlocal df
        # nonlocal low_indices
        
        # idx = low_indices.pop(0)
        # if np.std(series) == 0:
        #     return np.nan
        # elif np.std(series[:center]) == 0:
        #     return np.nan
        # elif np.std(series[center:]) == 0:
        #     return np.nan
        # # 변동성 조건
        # # elif any(np.abs(np.diff(series[:center]) / series[1:center]) * 100 >= 25.):
        # #     return np.nan
        # # 이평선 조건
        if series[center] == series.min():
            # if df.loc[idx, "5"] >  df.loc[idx, "20"] > df.loc[idx, "120"] and df.loc[idx, "Amount"] >= 50000000000:
            #     return series[center]
            # else:
            #     return np.nan
            return series[center]
        else:
            return np.nan
    
    # def rolling_wrapper(window, func):
    # # 각 윈도우의 끝 인덱스를 저장
    #     for i in window:
    #         high_indices.append(i.index[-1])
    #         low_indices.append(i.index[-1])
    #     return window.apply(func, raw=True)
    
    # df['fractal_high'] = rolling_wrapper(df["High"].rolling(window=n, center=True), fractal_high)
    # df['fractal_low'] = rolling_wrapper(df["Low"].rolling(window=n, center=True), fractal_low)
    df['fractal_high'] = df["High"].rolling(window=n, center=True).apply(fractal_high, raw=True)
    df['fractal_low'] = df["Low"].rolling(window=n, center=True).apply(fractal_low, raw=True)
    
    df.loc[~((df['5'] > df['20']) & (df['20'] > df['120'])), ['fractal_high', 'fractal_low']] = np.nan
    return df