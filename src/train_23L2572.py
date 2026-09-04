import pandas as pd
def load_data(path):
    try:
        print("Data Loading.....")
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None 

