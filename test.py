from scripts.data.dataset import read_data, create_dataset, load_dataset


import os
import pandas as pd
df = read_data(
    "data/test/20240618_선진뷰티사이언스.csv",
    pd.read_csv
    )