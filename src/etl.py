# ETL functions

import pandas as pd

def load_data(url):
    return pd.read_json(url, orient='records')
