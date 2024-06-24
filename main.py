import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout
from PyQt5.QtCore import pyqtSlot
from core.agent import Agent
from core.websocket import WebSocketWorker

from ui.search_bar import SearchBar
from ui.subscribe_table import SubscribeTable

import pandas as pd
import torch
from interface.models import CNN1D
class MyApp(QMainWindow):

    def __init__(self, agent, websocket_worker):
        super().__init__()
        self.agent = agent
        self.websocket_worker = websocket_worker
        
        self.setWindowTitle("WebSocket Client")
        self.setGeometry(100, 100, 600, 400)
        
        self.subscribe_table = SubscribeTable(self, websocket_worker.configure_transaction, websocket_worker.config_message_queue.put_nowait)
        self.search_bar = SearchBar(self, self.search)
    
        layout = QGridLayout()
        layout.addWidget(self.subscribe_table, 0, 0)
        layout.addWidget(self.search_bar, 1, 0)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.websocket_worker.message_received.connect(self.on_message_received)
        self.websocket_worker.start()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN1D(6, 3)
        self.model.load_state_dict(torch.load(f"models/saved/kis_day20_ma20120_cls3_5_epoch.pth"))
        self.model.to(self.device)
        self.model.eval()

    @pyqtSlot(list)
    def on_message_received(self, message):
        systime, code, time, price = message
        if code in self.subscribe_table.subscribe_list:
            if hasattr(self.subscribe_table.subscribe_list[code]["infer_thread"], "df"):
                self.subscribe_table.subscribe_list[code]["infer_thread"].live_recv.emit([time, price])   
            if hasattr(self.subscribe_table.subscribe_list[code]["infer_min"], "df"):
                self.subscribe_table.subscribe_list[code]["infer_min"].live_recv.emit([time, price])
        
    def search(self):
        if "KRW" in self.search_bar.name_input.text():
            code = self.search_bar.name_input.text()
            name = code
            if name in list(self.subscribe_table.subscribe_list.keys()):
                print("This code is already in the grid")
                return
            else:
                self.subscribe_table.add_subscribe(name, code)
                
        elif self.search_bar.code_input.text() == "" or self.search_bar.name_input.text() not in list(self.search_bar.names.keys()):
            print("Please enter a valid code")
            return
        else:
            code = self.search_bar.code_input.text()
            name = self.search_bar.name_input.text()
            
            if name in list(self.subscribe_table.subscribe_list.keys()):
                print("This code is already in the grid")
                return
            else:
                self.subscribe_table.add_subscribe(name, code)


if __name__ == '__main__':
    agent = Agent("simulation", "env.yaml")
    websocket_worker = WebSocketWorker(agent.name, agent.get_approval())
    app = QApplication(sys.argv)
    window = MyApp(agent, websocket_worker)
    window.show()
    sys.exit(app.exec_())