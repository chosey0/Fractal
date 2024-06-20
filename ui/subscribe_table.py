from PyQt5.QtWidgets import QTableWidget, QAbstractItemView, QHeaderView, QTableWidgetItem, QMenu, QAction
from PyQt5.QtCore import pyqtSignal, Qt
from ui.infer_plot import InferWindow

class SubscribeTable(QTableWidget):
    subscribe_signal = pyqtSignal(str, str)
    
    def __init__(self, parent, add_callback, subscribe_func):
        super().__init__(parent)
        self.parent = parent
        self.add_callback = add_callback
        self.subscribe_func = subscribe_func
        
        self.setRowCount(0)
        self.setColumnCount(1)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setHorizontalHeaderLabels(["웹소켓 연결 및 추론중인 종목"])
        
        self.subscribe_list = dict()
        self.cellDoubleClicked.connect(self.delete_row)
        
    def add_subscribe(self, name, code):
        tr_id, message, disconnect_message = self.add_callback(code)
        
        self.subscribe_list[code] = {
            "name": name, 
            "tr_id": tr_id, 
            "message": message, 
            "disconnect_message": disconnect_message, 
            "infer_window": InferWindow(self.parent.agent.day_candle, code, name)
            }
        
        self.setRowCount(len(self.subscribe_list))
        
        cell = QTableWidgetItem(name)
        cell.setData(Qt.UserRole, code)
        
        self.setItem(len(self.subscribe_list)-1, 0, cell)
        self.subscribe_func(message)
    
    def delete_row(self, row, column):
        
        code = self.item(row, column).data(Qt.UserRole)
        self.subscribe_func(self.subscribe_list[code]["disconnect_message"])
        del self.subscribe_list[code]
        self.removeRow(row)
        
    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        del_action = QAction('삭제', self)
        open_action = QAction('차트 열기', self)
        
        context_menu.addAction(del_action)
        context_menu.addAction(open_action)
        
        action = context_menu.exec_(self.mapToGlobal(event.pos()))
        
        if action == del_action:
            self.delete_row(self.currentRow(), self.currentColumn())
        if action == open_action:
            self.open_plot(self.currentRow(), self.currentColumn())
    
    def open_plot(self, row, column):
        code = self.item(row, column).data(Qt.UserRole)
        self.subscribe_list[code]["infer_window"].show()
        pass