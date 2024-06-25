import requests as req
import yaml
import json

import pandas as pd
import numpy as np
from core.api.ebest.auth import read_token

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
    try:
        res = req.post(URL, headers=headers, data=json.dumps(body))
    
        if res.json()["rsp_cd"] == 'IGW00121':
            headers["authorization"] = read_token("EBEST")
            res = req.post(URL, headers=headers, data=json.dumps(body))
            
        return res.json()["t8412OutBlock"], res.json()["t8412OutBlock1"], res.json()["rsp_cd"]
    except Exception as e:
        print(e)
        return None, None, None

def data_handler(data):

    df = pd.DataFrame(data)[["date", "time", "open", "high", "low", "close", "jdiff_vol"]]
    df["Time"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
    df.drop(["date", "time", "jdiff_vol"], axis=1, inplace=True)
    df.rename({"open": "Open", "high": "High", "low": "Low", "close": "Close"}, axis=1, inplace=True)
    df["ma20"] = df["Close"].rolling(window=20).mean()
    df["ma120"] = df["Close"].rolling(window=120).mean()
    df.dropna(inplace=True)
    
    df["low_point"] = [np.nan] * len(df)
    df["high_point"] = [np.nan] * len(df)
    df["low_prob"] = [np.nan] * len(df)
    df["high_prob"] = [np.nan] * len(df)
    df["none_prob"] = [np.nan] * len(df)
    df.set_index(pd.DatetimeIndex(df["Time"]).as_unit("ms").asi8, inplace=True)
    df[["Open", "Close", "High", "Low"]] = df[["Open", "Close", "High", "Low"]].astype(np.float64)
    return df[["Time", "Open", "Close", "High", "Low", "ma20", "ma120", "low_point", "high_point", "low_prob", "high_prob", "none_prob"]]
