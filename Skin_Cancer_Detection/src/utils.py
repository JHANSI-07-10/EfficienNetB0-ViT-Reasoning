import pandas as pd
from sklearn.model_selection import train_test_split

def get_data_splits(csv_path):
    df = pd.read_csv(csv_path)
    
    # Stratified split to keep the cancer percentage same in both sets
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['dx']
    )
    return train_df, val_df