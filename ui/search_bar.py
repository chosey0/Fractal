import sys

from PyQt5.QtWidgets import QWidget, QGridLayout, QLineEdit, QCompleter, QPushButton
from core.api.kosdaq import get_kosdaq_master_dataframe
from core.api.kospi import get_kospi_master_dataframe

import pandas as pd
import os

try:
    base_path = sys._MEIPASS
except Exception as e:
    base_path = os.getcwd()
    print(e)

class SearchBar(QWidget):
    def __init__(self, parent, return_action):
        super().__init__()
        self.parent = parent
        layout = QGridLayout()

        kosdaq = get_kosdaq_master_dataframe(os.path.join(base_path, "resource"))[["단축코드", "한글종목명"]]
        kospi = get_kospi_master_dataframe(os.path.join(base_path, "resource"))[["단축코드", "한글명"]].rename(columns={"한글명": "한글종목명"})

        # auto complete options
        self.df = pd.concat([kosdaq, kospi], axis=0)
        keys = self.df["한글종목명"].values
        values = self.df["단축코드"].values
        self.names = dict(zip(keys, values))

        completer = QCompleter(self.names.keys())

        # create line edits and add auto complete to the first one
        self.name_input = QLineEdit()
        self.name_input.setCompleter(completer)
        self.name_input.textChanged.connect(self.update_value)
        layout.addWidget(self.name_input, 0, 0)

        self.code_input = QLineEdit()
        self.code_input.setReadOnly(True)
        layout.addWidget(self.code_input, 0, 1)

        # create return button
        self.return_btn = QPushButton("return_btn")
        self.return_btn.clicked.connect(return_action)
        layout.addWidget(self.return_btn, 0, 2)
        
        self.setLayout(layout)

    def update_value(self, text):
        if text in self.names:
            self.code_input.setText(self.names[text])
        else:
            self.code_input.clear()