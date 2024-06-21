from PyQt5.QtWidgets import QGraphicsView, QGridLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
import finplot as fplt
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.nn import functional as F

import time
import pandas as pd
import numpy as np

from interface.models import CNN1D
from sklearn.preprocessing import minmax_scale, StandardScaler
import toolz.itertoolz as tz
from queue import Queue

fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = timezone.utc

class InferThread(QThread):
    init_data = pyqtSignal(object)
    data_update = pyqtSignal(object)
    live_recv = pyqtSignal(list)
    inference_return = pyqtSignal(object)
    
    def __init__(self, get_data_func, code) -> None:
        super().__init__()
        self.get_data_func = get_data_func
        self.code = code
        
        self.daemon = True
        self.data_queue = Queue()
        
        self.live_recv.connect(self.recv_datastr)
        
        self.scaler = StandardScaler()
        self.model = CNN1D(6, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.load_state_dict(torch.load(f"models/saved/kis_day20_ma20120_cls3_5_epoch.pth"))
        self.model.to(self.device)
        
    @pyqtSlot(list)
    def recv_datastr(self, recv_data):
        recv_time, datastr = recv_data
        recvstr = datastr.split('|')
        
        # 수신 데이터의 수량(나노 초 수준의 고빈도 데이터의 경우 여러건이 수신될 수 있음)
        n_item = int(recvstr[2])
        
        # 수신 데이터 전문
        data = recvstr[-1].split("^")
        
        if n_item == 1:
            new_data = [recv_time] + data
            # self.data_queue.put(new_data)
            self.calc_candle(new_data)
        else:
            temp = list(tz.partition(len(data)//n_item, data))
            chunks = [[(recv_time + idx*1e-3)]+list(chunk) for idx, chunk in enumerate(temp)]
            for chunk in chunks:
                # self.data_queue.put(chunk)
                self.calc_candle(chunk)
            self.inference(self.df.index[-1], self.df.iloc[-20:][["Open", "High", "Low", "Close", "ma20", "ma120"]])
    
    def calc_candle(self, data):
        dt = pd.to_datetime(str(datetime.now().date())+ " " + data[2], format='%Y-%m-%d %H%M%S').replace(hour=0, minute=0, second=0, microsecond=0)
        # dt = datetime.fromtimestamp(data[0]).replace(hour=0, minute=0, second=0, microsecond=0)
        price = np.array(data)[3].astype(np.float64)
        new_row = pd.DataFrame(data={"Time": dt, "Open": price, "Close": price, "High": price, "Low": price, "ma20": np.nan, "ma120": np.nan, 
                                        "low_point":np.nan, "high_point":np.nan}, index=pd.DatetimeIndex([dt]).as_unit("ms").asi8)

        if new_row.index[0] > self.df.index[-1]:
            self.df = pd.concat([self.df, new_row], axis=0)
        else:
            self.df.loc[new_row.index[0], "Close"] = price
            if self.df.iloc[-1]["High"] < price:
                self.df.loc[new_row.index[0],"High"] = price
            elif self.df.iloc[-1]["Low"] > price:
                self.df.loc[new_row.index[0], "Low"] = price
        self.df.loc[new_row.index[0], "ma20"] = self.df.iloc[-20:]["Close"].mean()
        self.df.loc[new_row.index[0], "ma120"] = self.df.iloc[-120:]["Close"].mean()
        
        self.data_update.emit(self.df[['Open', 'Close', 'High', 'Low', "ma20", "ma120", "low_point", "high_point"]])
        
    def run(self):
        self.df = get_previous_data(self.get_data_func, self.code, previous_data_handler)
        self.init_data.emit(self.df)
        
        with torch.no_grad():
            self.model.eval()
            self.prev_inference()
            self.live_inference()
    
    def prev_inference(self):
        for idx, dt in enumerate(self.df.index):
            data = self.df.loc[:dt, ["Open", "High", "Low", "Close", "ma20", "ma120"]]
            if data.empty or len(data) < 2:
                continue
            elif len(data) > 20:
                data = data.iloc[-20:]
                
            self.inference(dt, data)
            # self.inference_return.emit(self.df)

            time.sleep(0.001)
            
    def live_inference(self):
        while True:
            if not self.data_queue.empty():
                data = self.data_queue.get()
                
                dt = pd.to_datetime(str(datetime.now().date())+ " " + data[2], format='%Y-%m-%d %H%M%S').replace(hour=0, minute=0, second=0, microsecond=0)
                # dt = datetime.fromtimestamp(data[0]).replace(hour=0, minute=0, second=0, microsecond=0)
                price = np.array(data)[3].astype(np.float64)
                new_row = pd.DataFrame(data={"Time": dt, "Open": price, "Close": price, "High": price, "Low": price, "ma20": np.nan, "ma120": np.nan, 
                                             "low_point":np.nan, "high_point":np.nan}, index=pd.DatetimeIndex([dt]).as_unit("ms").asi8)

                if new_row.index[0] > self.df.index[-1]:
                    self.df = pd.concat([self.df, new_row], axis=0)
                else:
                    self.df.loc[new_row.index[0], "Close"] = price
                    if self.df.iloc[-1]["High"] < price:
                        self.df.loc[new_row.index[0],"High"] = price
                    elif self.df.iloc[-1]["Low"] > price:
                        self.df.loc[new_row.index[0], "Low"] = price
                self.df.loc[new_row.index[0], "ma20"] = self.df.iloc[-20:]["Close"].mean()
                self.df.loc[new_row.index[0], "ma120"] = self.df.iloc[-120:]["Close"].mean()
                
                self.data_update.emit(self.df[['Open', 'Close', 'High', 'Low', "ma20", "ma120", "low_point", "high_point"]])
                self.inference(new_row.index[0], self.df.iloc[-20:][["Open", "High", "Low", "Close", "ma20", "ma120"]])
                # self.inference_return.emit(self.df[["low_prob", "high_prob", "none_prob"]])
            else:
                time.sleep(0.001)
                
    @pyqtSlot(object)
    def inference(self, dt, data):
        input_data = torch.tensor(self.scaler.fit_transform(data.values), device=self.device, dtype=torch.float32)

        if len(input_data.size()) == 2:
            input_data = input_data.unsqueeze(0)
            
        logit = self.model(input_data)
        prob = F.softmax(logit, dim=1).flatten()
        
        self.df.loc[dt, "low_prob"] = prob[0].item()
        self.df.loc[dt, "high_prob"] = prob[1].item()
        self.df.loc[dt, "none_prob"] = prob[2].item()
        
        if 1. > prob[0] >= .95:
            self.df.loc[dt, "low_point"] = self.df.loc[dt, "Low"]
        elif 1. > prob[1] >= .95:
            self.df.loc[dt, "high_point"] = self.df.loc[dt, "High"]
        else:
            pass
        
class InferWindow(QWidget):
    def __init__(self, code, name):
        super().__init__()
        self.name = name
        self.code = code
        
        self.setWindowTitle(name)
        layout = QGridLayout()
        
        # self.infer_thread.init_data.connect(self.init_candle)
        # self.infer_thread.data_update.connect(self.update_candle)
        # # self.infer_thread.inference_return.connect(self.update_prob)
        # self.infer_thread.start()
        
        self.setWindowTitle("QGraphicsView")
        
        self.resize(1024, 680)
        
        self.view = QGraphicsView()
        self.view_layout = QGridLayout()
        self.view.setLayout(self.view_layout)
        # ax
        # ax0, ax1 = fplt.create_plot(init_zoom_periods=100, rows=2)
        # self.axs = [ax0, ax1] # finplot requres this property
        
        # for idx, ax in enumerate(self.axs):
        #     self.view_layout.addWidget(ax.vb.win, idx, 0) 
        self.ax = fplt.create_plot(init_zoom_periods=100, rows=1)
        self.view_layout.addWidget(self.ax.vb.win, 0, 0) 
           
        layout.addWidget(self.view)
        
        self.setLayout(layout)
    
    @pyqtSlot(object)
    def update_candle(self, df):

        self.candle.update_data(df[['Open', 'Close', 'High', 'Low']])
        self.ma20.update_data(df["ma20"])
        self.ma120.update_data(df["ma120"])
        self.low_point.update_data(df["low_point"])
        self.high_point.update_data(df["high_point"])
        
    @pyqtSlot(object)
    def update_prob(self, df):
        self.low_prob.update_data(df[f"low_prob"])
        self.high_prob.update_data(df[f"high_prob"])
        self.none_prob.update_data(df[f"none_prob"])

    @pyqtSlot(object)
    def init_candle(self, df):
        # ax0, ax1 = self.axs
        self.candle = fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=self.ax)
        self.ma20 = fplt.plot(df["ma20"], color='#5A639C', width=2.0, ax=self.ax)
        self.ma120 = fplt.plot(df["ma120"], color='#A0937D', width=2.0, ax=self.ax)
        self.low_point = fplt.plot(df["low_point"], style="o", color='#00ffbc', width=2.0, ax=self.ax)
        self.high_point = fplt.plot(df["high_point"], style="o", color='#ffbcff', width=2.0, ax=self.ax)
        
        # self.low_prob = fplt.plot(df[f"low_prob"], style="o", color="#973131", width=1, ax=ax1)
        # self.high_prob = fplt.plot(df[f"high_prob"], style="o", color="#5A639C", width=1, ax=ax1)
        # self.none_prob = fplt.plot(df[f"none_prob"], style="o", color="#A0937D", width=1, ax=ax1)
        fplt.show(qt_exec=False)


def get_previous_data(get_data_func, code, data_handler):
    prev_data = []
    FID_INPUT_DATE_2 = datetime.now()
    res = get_data_func(code, FID_INPUT_DATE_2=FID_INPUT_DATE_2)
    prev_data.extend(res.json()["output2"][:-1])
    
    for _ in range(5):
        time.sleep(0.5)
        FID_INPUT_DATE_2 = datetime.strptime(res.json()["output2"][-1]["stck_bsop_date"], "%Y%m%d")
        res = get_data_func(code, FID_INPUT_DATE_2=FID_INPUT_DATE_2)
        prev_data.extend(res.json()["output2"][:-1])
        
    return data_handler(prev_data)

def previous_data_handler(data):
    df = pd.DataFrame(data)[::-1].reset_index(drop=True)
    df.rename({"stck_bsop_date": "Time", "stck_clpr": "Close", "stck_oprc": "Open", "stck_hgpr": "High", "stck_lwpr": "Low"}, axis=1, inplace=True)
    
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(np.int64)
    
    df["ma20"] = df["Close"].rolling(window=20).mean()
    df["ma120"] = df["Close"].rolling(window=120).mean()
    df["Time"] = pd.to_datetime(df['Time'], format='%Y%m%d')
    
    df.set_index(pd.DatetimeIndex(df["Time"]).as_unit("ms").asi8, inplace=True)
    
    df = df[["Time", "Open", "High", "Low", "Close", "ma20", "ma120"]].dropna()
    
    df["low_point"] = [np.nan] * len(df)
    df["high_point"] = [np.nan] * len(df)
    df["low_prob"] = [np.nan] * len(df)
    df["high_prob"] = [np.nan] * len(df)
    df["none_prob"] = [np.nan] * len(df)
    
    return df