import pandas as pd
from file_utils import load_and_analyze_dataset, path

def main():
    print("🧠 Epileptic Seizure Recognition Dataset Analysis")
    print("=" * 50)
    
    # Load and analyze the dataset
    X, y = load_and_analyze_dataset(path, target_column='y')
    
    if X is not None and y is not None:
        print("\n Dataset loaded successfully!")
        print(f"Total samples: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Unique labels: {sorted(y.unique())}")
        print(f"Label distribution:")
        print(y.value_counts().sort_index())
    else:
        print("\n Failed to load dataset")

if __name__ == "__main__":
    main()
