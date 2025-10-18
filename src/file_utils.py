import pandas as pd 
import os

path = 'data/epileptic_seizure_recognition.csv'

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def split_features_labels(df, target_column=None):
    
    if target_column is None:
        target_column = df.columns[-1]
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def load_and_split_data(file_path, target_column=None):
    df = read_csv(file_path)
    return split_features_labels(df, target_column)

def load_and_analyze_dataset(dataset_path, target_column='y'):
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found: {dataset_path}")
        X, y = load_and_split_data(dataset_path, target_column=target_column)
        
        print(f"Features shape (X): {X.shape}")
        print(f"Labels shape (y): {y.shape}")
        print("\nFirst 5 rows of features:")
        print(X.head())
        print("\nFirst 5 labels:")
        print(y.head())
        
        return X, y
    else:
        print(f"Error: Dataset not found at path: {dataset_path}")
        return None, None

