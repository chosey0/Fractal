from tqdm import tqdm
from PyQt5.QtWidgets import QTableWidget, QAbstractItemView, QHeaderView, QTableWidgetItem
from PyQt5.QtCore import pyqtSignal, pyqtSlot

import threading
from queue import Queue

import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

import time
from datetime import datetime

from core.api.ebest.tickChart import TickRequest, data_handler

class InferMinWindow(QTableWidget):
    live_recv = pyqtSignal(list)
    
    def __init__(self, model, code, name):
        super().__init__()
        self.name = name
        self.code = code
        
        self.recv_queue = Queue()
        self.live_queue = Queue()
        self.live_recv.connect(self.recv_datastr)
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
        
        self.df = None
        
        threading.Thread(target=self.get_previous_min_candle, name=f"{self.name} Prev Inference", args=(code,)).start()
        threading.Thread(target=self.live_inference, name=f"{self.name} Live Inference", ).start()
        threading.Thread(target=self.update_candle, name=f"{self.name} candle Updator", ).start()
        
    @pyqtSlot(list)
    def recv_datastr(self, message):
        self.recv_queue.put(message)
    
    def update_candle(self):
        while True:
            if not self.recv_queue.empty() and self.df is not None:
                dt_str, price = self.recv_queue.get()
                dt = pd.to_datetime(str(datetime.now().date())+ " " + dt_str, format='%Y-%m-%d %H%M%S').replace(second=0, microsecond=0)
            
                idx_dt = pd.DatetimeIndex([dt]).as_unit("ms").asi8[0]

                try:
                    if not idx_dt in self.df.index:
                        new_row = pd.DataFrame(data={"Time": dt, "Open": price, "Close": price, "High": price, "Low": price, "ma20": np.nan, "ma120": np.nan, "low_point": np.nan, "high_point": np.nan, "low_prob": np.nan, "high_prob": np.nan, "none_prob": np.nan}, index=[idx_dt])
                        self.df = pd.concat([self.df, new_row])
                        if self.df.shape[0] > 20:
                            self.live_queue.put([idx_dt, self.df.loc[self.df.index[-20]:, ["Open", "High", "Low", "Close", "ma20", "ma120"]].values])
                            
                    elif idx_dt in self.df.index:
                        if self.df.loc[idx_dt, "Close"] != price:
                            self.df.loc[idx_dt, "Close"] = price
                            if self.df.shape[0] > 20:
                                self.df.loc[idx_dt, "ma20"] = self.df.loc[self.df.index[-20]:]["Close"].mean()
                            if self.df.shape[0] > 120:
                                self.df.loc[idx_dt, "ma120"] = self.df.loc[self.df.index[-120]:]["Close"].mean()
                        elif self.df.loc[idx_dt, "High"] < price:
                            self.df.loc[idx_dt, "High"] = price
                        elif self.df.loc[idx_dt, "Low"] > price:
                            self.df.loc[idx_dt, "Low"] = price
                        else:
                            continue
                        
                        if self.df.shape[0] > 20:
                            self.live_queue.put([idx_dt, self.df.loc[self.df.index[-20]:][["Open", "High", "Low", "Close", "ma20", "ma120"]].values])
                except Exception as e:
                    print(idx_dt)
                    print(new_row)
                    print(self.df.head())
                    print(e)
            else:
                time.sleep(0.001)
    
    def add_low_point(self, time, low_prob, high_prob, other_prob):
            self.low_points[time] = [low_prob, high_prob, other_prob]
            
            if self.item(0, 0) is not None:
                if datetime.strptime(self.item(0, 0).text(), "%Y-%m-%d %H:%M:%S") == datetime.strptime(time.isoformat(), "%Y-%m-%dT%H:%M:%S"):
                    self.setItem(0, 1, QTableWidgetItem(str(round(float(low_prob), 4))))
                    self.setItem(0, 2, QTableWidgetItem(str(round(float(high_prob), 4))))
                    self.setItem(0, 3, QTableWidgetItem(str(round(float(other_prob), 4))))
                    return

            self.setRowCount(len(self.low_points))
            row = 0
            self.insertRow(row)
            
            time_cell = QTableWidgetItem(time.strftime("%Y-%m-%d %H:%M:%S"))
            low_cell = QTableWidgetItem(str(round(float(low_prob), 4)))
            high_cell = QTableWidgetItem(str(round(float(high_prob), 4)))
            other_cell = QTableWidgetItem(str(round(float(other_prob), 4)))
            
            self.setItem(row, 0, time_cell)
            self.setItem(row, 1, low_cell)
            self.setItem(row, 2, high_cell)
            self.setItem(row, 3, other_cell)
            
    def live_inference(self):
        with torch.no_grad():
            while True:
                if self.df is not None and not self.live_queue.empty():
                    dt, data = self.live_queue.get()
                    input_data = torch.tensor(self.scaler.fit_transform(data), device=self.device, dtype=torch.float32)
                    timestr = self.df.loc[dt, "Time"]
                    if len(input_data.size()) == 2:
                        input_data = input_data.unsqueeze(0)
                        
                    logit = self.model(input_data)
                    prob = F.softmax(logit, dim=1).flatten()
                    
                    self.df.loc[dt, "low_prob"] = prob[0].item()
                    self.df.loc[dt, "high_prob"] = prob[1].item()
                    self.df.loc[dt, "none_prob"] = prob[2].item()

                    if prob[0] >= .95:
                        self.add_low_point(timestr, prob[0].item(), prob[1].item(), prob[2].item())
                        print(f"Probabilities: {prob}")
                    else:
                        continue
                else:
                    time.sleep(0.001)
                    
    def prev_inference(self):
        with torch.no_grad():
            for idx, dt in tqdm(enumerate(self.df.index), total=len(self.df),desc=f"{self.name} prev inference"):
                
                data = self.df.loc[:dt, ["Open", "High", "Low", "Close", "ma20", "ma120"]].values
                timestr = self.df.loc[dt, "Time"]
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
                    self.add_low_point(timestr, prob[0].item(), prob[1].item(), prob[2].item())
                else:
                    continue

    def get_previous_min_candle(self, code):
        
        prev_data = []
        meta, data, cont_key = TickRequest("EBEST", code, 1, 500)
        prev_data = [*data, *prev_data]
        try:
            for _ in range(3):
                time.sleep(1)
                meta, data, cont_key = TickRequest("EBEST", code, 1, 500, cts_date=meta["cts_date"], cts_time=meta["cts_time"], tr_cont="Y", tr_cont_key=cont_key)
                prev_data = [*data[:-1], *prev_data]
        except:
            pass
        self.df = data_handler(prev_data)
        return threading.Thread(target=self.prev_inference, name=f"{self.name} Prev Inference").start()

