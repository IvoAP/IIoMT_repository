import os
import pandas as pd

# Default dataset locations
path = 'data/epileptic_seizure_recognition.csv'
binary_path = 'data/epileptic_seizure_recognition_binary.csv'


def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def split_features_labels(df: pd.DataFrame, target_column: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if target_column is None:
        target_column = df.columns[-1]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def load_and_split_data(file_path: str, target_column: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    df = read_csv(file_path)
    return split_features_labels(df, target_column)


def generate_binary_dataset(
    original_dataset_path: str = path,
    binary_dataset_path: str = binary_path,
    target_column: str = 'y',
    positive_class: int = 1,
) -> str:
    """Generate and persist a binary (seizure vs non-seizure) dataset."""
    if not os.path.exists(original_dataset_path):
        raise FileNotFoundError(f"Original dataset not found at: {original_dataset_path}")

    df = read_csv(original_dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    binary_df = df.copy()
    binary_df[target_column] = (binary_df[target_column] == positive_class).astype(int)

    os.makedirs(os.path.dirname(binary_dataset_path), exist_ok=True)
    binary_df.to_csv(binary_dataset_path, index=False)

    return binary_dataset_path


def ensure_binary_dataset(
    original_dataset_path: str = path,
    binary_dataset_path: str = binary_path,
    target_column: str = 'y',
    positive_class: int = 1,
) -> str:
    """Ensure a binary dataset exists, generating it if necessary."""
    if not os.path.exists(binary_dataset_path):
        generate_binary_dataset(original_dataset_path, binary_dataset_path, target_column, positive_class)

    return binary_dataset_path


def load_and_analyze_dataset(
    dataset_path: str,
    target_column: str = 'y',
    dataset_name: str | None = None,
    class_names: dict[int, str] | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Load a dataset, print a quick analysis, and return X/y."""
    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found at {dataset_path}")
        return None, None

    if dataset_name is None:
        dataset_name = os.path.basename(dataset_path)

    X, y = load_and_split_data(dataset_path, target_column=target_column)

    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Target: {target_column}")

    label_counts = y.value_counts().sort_index()
    print("Label distribution:")
    for label, count in label_counts.items():
        label_name = class_names.get(label, label) if class_names else label
        percent = count / len(y) * 100
        print(f"  {label_name}: {count} ({percent:.2f}%)")

    return X, y

