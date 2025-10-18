import os
from file_utils import (
    binary_path,
    ensure_binary_dataset,
    load_and_analyze_dataset,
    path,
)

ORIGINAL_CLASS_NAMES = {
    1: 'Seizure',
    2: 'TumorRegion',
    3: 'HealthyRegion',
    4: 'EyesClosed',
    5: 'EyesOpen',
}

BINARY_CLASS_NAMES = {
    0: 'Non-Seizure',
    1: 'Seizure',
}

def main():
    print("Epileptic Seizure Recognition Dataset Analysis")

    if not os.path.exists(path):
        print(f"Original dataset not found at {path}")
        return

    print("\nOriginal 5-class dataset:")
    load_and_analyze_dataset(
        dataset_path=path,
        target_column='y',
        dataset_name='Epileptic Seizure Recognition (5-class)',
        class_names=ORIGINAL_CLASS_NAMES,
    )

    binary_already_exists = os.path.exists(binary_path)
    ensured_binary_path = ensure_binary_dataset(path, binary_path)
    if not binary_already_exists:
        print(f"\nBinary dataset generated at {ensured_binary_path}")

    print("\nBinary dataset:")
    load_and_analyze_dataset(
        dataset_path=ensured_binary_path,
        target_column='y',
        dataset_name='Epileptic Seizure Recognition (Binary)',
        class_names=BINARY_CLASS_NAMES,
    )

if __name__ == "__main__":
    main()
