import numpy as np
import pandas as pd

def calc_fractal(df: pd.DataFrame, n=500):
    # 프랙탈을 계산할 기간 (일반적으로 5일을 사용)

    high_indices = []
    low_indices = []
    def fractal_high(series):
        center = n // 2
        if series[center] == series.max():
            return series[center]
        else:
            return np.nan

    def fractal_low(series):
        center = n // 2
        # 이평선 조건
        if series[center] == series.min():

            return series[center]
        else:
            return np.nan

    df['fractal_high'] = df["High"].rolling(window=n, center=True).apply(fractal_high, raw=True)
    df['fractal_low'] = df["Low"].rolling(window=n, center=True).apply(fractal_low, raw=True)
    
    # df.loc[~(df['20'] > df['120']), ['fractal_high', 'fractal_low']] = np.nan
    # df.loc[~((df['5'] > df['20']) & (df['20'] > df['120'])), ['fractal_high', 'fractal_low']] = np.nan
    # df.loc[df["Amount"] < 500000000, ['fractal_high', 'fractal_low']] = np.nan
    return df