import numpy as np
import pandas as pd

def calc_fractal(df: pd.DataFrame, n=500):
    # 프랙탈을 계산할 기간 (일반적으로 5일을 사용)

    def fractal_high(series):
        center = n // 2
        # 변동 계수
        # window_std = np.std(series)
        # window_mean = np.mean(series)
        # window_cv = window_std / window_mean * 100
        
        # kernel_std = np.std(series[center:])
        # kernel_mean = np.mean(series[center:])
        # kernel_cv = kernel_std / kernel_mean * 100
        
        # print(f"window: {window_cv}, kernel: {kernel_cv}")
        # if window_cv < 2:
        #     return np.nan
        if np.std(series) == 0:
            return np.nan
        elif np.std(series[:center]) == 0:
            return np.nan
        elif np.std(series[center:]) == 0:
            return np.nan
        elif series[center] == series.max():
            return series[center]
        else:
            return np.nan

    def fractal_low(series):
        center = n // 2
        
        # window_std = np.std(series)
        # window_mean = np.mean(series)
        # window_cv = window_std / window_mean * 100
        
        # kernel_std = np.std(series[center:])
        # kernel_mean = np.mean(series[center:])
        # kernel_cv = kernel_std / kernel_mean * 100
        
        # print(f"window: {window_cv}, kernel: {kernel_cv}")
        # if window_cv < 2:
        #     return np.nan
        if np.std(series) == 0:
            return np.nan
        elif np.std(series[:center]) == 0:
            return np.nan
        elif np.std(series[center:]) == 0:
            return np.nan
        elif series[center] == series.min():
            return series[center]
        else:
            return np.nan

    df['fractal_high'] = df['High'].rolling(window=n, center=True).apply(fractal_high, raw=True)
    df['fractal_low'] = df["Low"].rolling(window=n, center=True).apply(fractal_low, raw=True)
    return df