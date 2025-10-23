import os
from typing import Any

from . import BaseRunner
from ..file_utils import generate_binary_dataset, ensure_binary_dataset, path, binary_path


class BinaryDatasetRunner(BaseRunner):
    """Runner for generating and managing binary datasets."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.original_dataset_path = self.config.get('original_dataset_path', path)
        self.binary_dataset_path = self.config.get('binary_dataset_path', binary_path)
        self.force_regenerate = self.config.get('force_regenerate', False)
    
    def run(self) -> None:
        """Generate or ensure binary dataset exists."""
        print(f"Binary Dataset Runner")
        print(f"Original dataset: {self.original_dataset_path}")
        print(f"Binary dataset: {self.binary_dataset_path}")
        
        if not os.path.exists(self.original_dataset_path):
            raise FileNotFoundError(f"Original dataset not found at {self.original_dataset_path}")
        
        if self.force_regenerate or not os.path.exists(self.binary_dataset_path):
            print("Generating binary dataset...")
            result_path = generate_binary_dataset(
                original_dataset_path=self.original_dataset_path,
                binary_dataset_path=self.binary_dataset_path
            )
            print(f"Binary dataset generated successfully at: {result_path}")
        else:
            print("Binary dataset already exists. Ensuring integrity...")
            ensure_binary_dataset(self.original_dataset_path, self.binary_dataset_path)
            print("Binary dataset integrity verified.")
        
        print(f"Binary dataset is ready at: {self.binary_dataset_path}")


def create_binary_dataset_runner(
    original_dataset_path: str | None = None,
    binary_dataset_path: str | None = None,
    force_regenerate: bool = False
) -> BinaryDatasetRunner:
    """Factory function to create a binary dataset runner."""
    config: dict[str, Any] = {
        'original_dataset_path': original_dataset_path or path,
        'binary_dataset_path': binary_dataset_path or binary_path,
        'force_regenerate': force_regenerate
    }
    return BinaryDatasetRunner(config)


if __name__ == "__main__":
    # Example usage
    runner = create_binary_dataset_runner(force_regenerate=True)
    runner.execute()