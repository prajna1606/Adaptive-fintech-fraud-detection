import pandas as pd

def load_transactions(path):
    df = pd.read_csv(path)
    return df
