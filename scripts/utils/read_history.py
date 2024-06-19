import pickle
import pandas as pd

def read_history(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    history_df = pd.DataFrame(data)
    return history_df