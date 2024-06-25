from PyQt5.QtWidgets import QTableWidget, QAbstractItemView, QHeaderView, QTableWidgetItem, QMenu, QAction
from PyQt5.QtCore import pyqtSignal, Qt
# 일봉
from ui.infer_plot import InferWindow, InferThread
# 분봉
from ui.infer_plot_min import InferMinWindow

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
        # self.cellDoubleClicked.connect(self.delete_row)

    def add_subscribe(self, name, code):
        tr_id, message, disconnect_message = self.add_callback(code)
        
        infer_window = InferWindow(code, name)
        infer_thread = InferThread(self.parent.model, self.parent.agent.day_candle, code, name)
        
        infer_thread.init_candle.connect(infer_window.init_candle)
        infer_thread.update_candle.connect(infer_window.update_candle)
        infer_thread.update_prob.connect(infer_window.update_prob)
        
        self.subscribe_list[code] = {
            "name": name, 
            "tr_id": tr_id, 
            "message": message, 
            "disconnect_message": disconnect_message, 
            "infer_window": infer_window,
            "infer_thread": infer_thread,
            "infer_min": InferMinWindow(self.parent.model, code, name)
            }
        
        self.subscribe_list[code]["infer_thread"].start()
        self.setRowCount(len(self.subscribe_list))
        
        cell = QTableWidgetItem(name)
        cell.setData(Qt.UserRole, code)
        
        self.setItem(len(self.subscribe_list)-1, 0, cell)
        self.subscribe_func(message)
    
    def delete_row(self, row, column):
        
        code = self.item(row, column).data(Qt.UserRole)
        self.subscribe_func(self.subscribe_list[code]["disconnect_message"])
        
        self.subscribe_list[code]["infer_thread"].quit()
        self.subscribe_list[code]["infer_window"].close()
        del self.subscribe_list[code]

        self.removeRow(row)
        
    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        del_action = QAction('삭제', self)
        open_action = QAction('차트 열기', self)
        min_action = QAction('분봉', self)
        
        context_menu.addAction(del_action)
        context_menu.addAction(open_action)
        context_menu.addAction(min_action)
        
        action = context_menu.exec_(self.mapToGlobal(event.pos()))
        
        if action == del_action:
            self.delete_row(self.currentRow(), self.currentColumn())
        if action == open_action:
            self.open_plot(self.currentRow(), self.currentColumn())
        if action == min_action:
            self.open_min_table(self.currentRow(), self.currentColumn())
    
    def open_plot(self, row, column):
        code = self.item(row, column).data(Qt.UserRole)
        self.subscribe_list[code]["infer_window"].show()
        
    def open_min_table(self, row, column):
        code = self.item(row, column).data(Qt.UserRole)
        self.subscribe_list[code]["infer_min"].show()