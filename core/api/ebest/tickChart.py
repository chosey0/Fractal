import requests as req
import yaml
import json

import pandas as pd
import time
import numpy as np
def TickRequest(name,
                stockcode, 
                size: int, 
                n_data: int, edate="99999999", cts_date="", cts_time="", tr_cont="N", tr_cont_key=''):
    with open("env.yaml", encoding='UTF-8') as f:
        config = yaml.safe_load(f)[name]
        
    prev_data = []
    URL= 'https://openapi.ls-sec.co.kr:8080/stock/chart'
    TOKEN = config["TOKEN"]
    
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "authorization": TOKEN,
        "tr_cd": 't8412',
        "tr_cont": tr_cont,
        "tr_cont_key": tr_cont_key
    }
    
    body = {
        "t8412InBlock": {
            "shcode": stockcode,
            "ncnt": size,
            "qrycnt": n_data,
            "nday": '0',
            "sdate": '',
            "stime": '',
            "edate": edate,
            "etime": '',
            "cts_date": cts_date,
            "cts_time": cts_time,
            "comp_yn": 'N'
        }
    }
    
    res = req.post(URL, headers=headers, data=json.dumps(body))
    return res.json()["t8412OutBlock"], res.json()["t8412OutBlock1"], res.json()["rsp_cd"]

def data_handler(data):
    df = pd.DataFrame(data)[["date", "time", "open", "high", "low", "close", "jdiff_vol"]]
    df["Time"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
    df.drop(["date", "time"], axis=1, inplace=True)
    df.rename({"open": "Open", "high": "High", "low": "Low", "close": "Close", "jdiff_vol": "Volume"}, axis=1, inplace=True)
    df["ma20"] = df["Close"].rolling(window=20).mean()
    df["ma120"] = df["Close"].rolling(window=120).mean()
    df.dropna(inplace=True)
    
    df["low_point"] = [np.nan] * len(df)
    df["high_point"] = [np.nan] * len(df)
    df["low_prob"] = [np.nan] * len(df)
    df["high_prob"] = [np.nan] * len(df)
    df["none_prob"] = [np.nan] * len(df)
    df.set_index(pd.DatetimeIndex(df["Time"]).as_unit("ms").asi8, inplace=True)
    return df
