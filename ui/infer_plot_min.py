from PyQt5.QtWidgets import QGraphicsView, QGridLayout, QWidget, QFileDialog
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
import cupy as cp

from core.api.ebest.tickChart import TickRequest, data_handler
fplt.candle_bull_color = "#ffbbc0"
fplt.candle_bull_body_color = "#ffbbc0" 
fplt.candle_bear_color = "#bbc0ff"
fplt.candle_bear_body_color = "#bbc0ff"
fplt.display_timezone = timezone.utc

from tqdm import tqdm
from PyQt5.QtWidgets import QTableWidget, QAbstractItemView, QHeaderView, QTableWidgetItem, QMenu, QAction
from PyQt5.QtCore import pyqtSignal, Qt

import threading

class InferMinWindow(QTableWidget):
    live_recv = pyqtSignal(list)
    
    def __init__(self, model, code, name):
        super().__init__()
        self.name = name
        self.code = code
        
        self.recv_queue = Queue()
        self.live_queue = Queue()
        self.live_recv.connect(self.calc_candle)
        self.setWindowTitle(name)
        self.setRowCount(1)
        self.setColumnCount(4)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setHorizontalHeaderLabels(["Time", "Low 확률", "High 확률", "Other 확률"])
        
        self.low_points = dict()
        self.resize(680, 1024)
        
        self.scaler = StandardScaler()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.df = self.get_previous_min_candle(code)
        
        threading.Thread(target=self.prev_inference).start()
        
    
    @pyqtSlot(list)
    def calc_candle(self, message):
        time, price_str = message
        dt = pd.to_datetime(str(datetime.now().date())+ " " + time, format='%Y-%m-%d %H%M%S').replace(second=0, microsecond=0)
        
        idx_dt = pd.DatetimeIndex([dt]).as_unit("ms").asi8[0]
        price = float(price_str)
        
        if not idx_dt in self.df.index:
            new_row = pd.DataFrame(data={"Time": dt, "Open": price, "Close": price, "High": price, "Low": price, "ma20": np.nan, "ma120": np.nan, "low_prob": np.nan, "high_prob": np.nan, "none_prob": np.nan}, index=pd.DatetimeIndex([dt]).as_unit("ms").asi8)
            if not hasattr(self, "df"):
                self.df = new_row
                return
            else:
                self.df.loc[idx_dt] = new_row
                if self.df.shape[0] > 120:
                    self.live_queue.put([idx_dt, self.df.iloc[-20:, ["Open", "High", "Low", "Close", "ma20", "ma120"]].values])
                
        elif idx_dt in self.df.index:
            if self.df.loc[idx_dt, "Close"] != price:
                self.df.loc[idx_dt, "Close"] = price
                if self.df.shape[0] > 20:
                    self.df.loc[idx_dt, "ma20"] = self.df.iloc[-20:]["Close"].mean()
                if self.df.shape[0] > 120:
                    self.df.loc[idx_dt, "ma120"] = self.df.iloc[-120:]["Close"].mean()
            elif self.df.loc[idx_dt, "High"] < price:
                self.df.loc[idx_dt, "High"] = price
            elif self.df.loc[idx_dt, "Low"] > price:
                self.df.loc[idx_dt, "Low"] = price
            else:
                return
            
            if self.df.shape[0] > 120:
                self.live_queue.put([idx_dt, self.df.iloc[-20:, ["Open", "High", "Low", "Close", "ma20", "ma120"]].values])
    
    def add_low_point(self, time, low_prob, high_prob, other_prob):
            
            self.low_points[time] = [low_prob, high_prob, other_prob]
            self.setRowCount(len(self.low_points))
            row = len(self.low_points)-1
            
            time_cell = QTableWidgetItem(time.strftime("%Y-%m-%d %H:%M:%S"))
            low_cell = QTableWidgetItem(str(round(float(low_prob), 4)))
            high_cell = QTableWidgetItem(str(round(float(high_prob), 4)))
            other_cell = QTableWidgetItem(str(round(float(other_prob), 4)))
            
            self.setItem(row, 0, time_cell)
            self.setItem(row, 1, low_cell)
            self.setItem(row, 2, high_cell)
            self.setItem(row, 3, other_cell)
            
    def update_low_point(self, time, low_prob, high_prob, other_prob):
        if time in self.low_points:
            self.low_points[time] = [low_prob, high_prob, other_prob]
            row = [i for i in range(self.rowCount()) if self.item(i, 0).text() == time][0]
            
            low_cell = QTableWidgetItem(str(round(float(low_prob), 4)))
            high_cell = QTableWidgetItem(str(round(float(high_prob), 4)))
            other_cell = QTableWidgetItem(str(round(float(other_prob), 4)))
            
            self.setItem(row, 1, low_cell)
            self.setItem(row, 2, high_cell)
            self.setItem(row, 3, other_cell)
            
        else:
            self.add_low_point(time, low_prob, high_prob, other_prob)
            
    def live_inference(self):
        with torch.no_grad():
            while True:
                if not self.live_queue.empty():
                    dt, data = self.live_queue.get()
                    input_data = torch.tensor(self.scaler.fit_transform(data), device=self.device, dtype=torch.float32)
            
                    if len(input_data.size()) == 2:
                        input_data = input_data.unsqueeze(0)
                        
                    logit = self.model(input_data)
                    prob = F.softmax(logit, dim=1).flatten()
                    
                    self.df.loc[dt, "low_prob"] = prob[0].item()
                    self.df.loc[dt, "high_prob"] = prob[1].item()
                    self.df.loc[dt, "none_prob"] = prob[2].item()
                    
                    if prob[0] >= .95:
                        self.add_low_point(self.df.loc[dt, "Time"], prob[0].item(), prob[1].item(), prob[2].item())
                    else:
                        pass
                else:
                    time.sleep(0.001)
                    
    def prev_inference(self):
        with torch.no_grad():
            for idx, dt in tqdm(enumerate(self.df.index), total=len(self.df),desc=f"{self.name} prev inference"):
                
                data = self.df.loc[:dt, ["Open", "High", "Low", "Close", "ma20", "ma120"]].values

                if len(data) < 2:
                    continue
                
                elif len(data) > 20:
                    data = data[-20:, :]
                    
                input_data = torch.tensor(self.scaler.fit_transform(data), device=self.device, dtype=torch.float32)
                
                if len(input_data.size()) == 2:
                    input_data = input_data.unsqueeze(0)
                
                logit = self.model(input_data)
                prob = F.softmax(logit, dim=1).flatten()
                
                if prob[0].item() >= .95:
                    self.add_low_point(self.df.loc[dt, "Time"], prob[0].item(), prob[1].item(), prob[2].item())
                else:
                    continue
                
        return threading.Thread(target=self.live_inference).start()
                
    def get_previous_min_candle(self, code):
        try:
            prev_data = []
            meta, data, cont_key = TickRequest("EBEST", code, 1, 500)
            prev_data = [*data, *prev_data]
            for _ in range(3):
                time.sleep(1)
                meta, data, cont_key = TickRequest("EBEST", code, 1, 500, cts_date=meta["cts_date"], cts_time=meta["cts_time"], tr_cont="Y", tr_cont_key=cont_key)
                prev_data = [*data[:-1], *prev_data]
                
            return data_handler(prev_data)
        except Exception as e:  
            print(f"Failed to load data: {e}, self.get_previous_min_candle")
    
    
    
    def previous_data_handler(self, df, format):
        
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(np.int64)
        
        df["ma20"] = df["Close"].rolling(window=20).mean()
        df["ma120"] = df["Close"].rolling(window=120).mean()
        df["Time"] = pd.to_datetime(df['Time'], format=format)
        
        df.set_index(pd.DatetimeIndex(df["Time"]).as_unit("ms").asi8, inplace=True)
        
        df = df[["Time", "Open", "High", "Low", "Close", "ma20", "ma120"]].dropna()
        
        df["low_prob"] = [np.nan] * len(df)
        df["high_prob"] = [np.nan] * len(df)
        df["none_prob"] = [np.nan] * len(df)

        return df.iloc[-200:]

