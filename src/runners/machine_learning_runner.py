import os
from typing import Any

from . import BaseRunner
from ..constraints import BINARY_CLASS_NAMES, ORIGINAL_CLASS_NAMES
from ..file_utils import (
    binary_path, 
    ensure_binary_dataset, 
    generate_binary_dataset,
    path
)
from ..ml.training import train_and_evaluate


class MachineLearningRunner(BaseRunner):
    """Runner for machine learning experiments."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.dataset_keys = self.config.get('dataset_keys', ['five', 'binary'])
        self.regenerate_binary = self.config.get('regenerate_binary', False)
        self.classical_trials = self.config.get('classical_trials', None)
        self.dnn_trials = self.config.get('dnn_trials', None)
        self.verbose = self.config.get('verbose', True)
        
        # Ensure ML results directory
        self.ml_dir = self.results_dir / "ml"
        self.ml_dir.mkdir(exist_ok=True)
    
    def _get_dataset_metadata(self, dataset_key: str) -> tuple[str, str, dict[int, str]]:
        """Get metadata for a specific dataset."""
        if dataset_key == "five":
            return path, "Epileptic Seizure Recognition (5-class)", ORIGINAL_CLASS_NAMES
        if dataset_key == "binary":
            return binary_path, "Epileptic Seizure Recognition (Binary)", BINARY_CLASS_NAMES
        raise ValueError(f"Unsupported dataset key: {dataset_key}")
    
    def _prepare_binary_dataset(self) -> str:
        """Prepare binary dataset if needed."""
        if self.regenerate_binary or not os.path.exists(binary_path):
            return generate_binary_dataset(original_dataset_path=path, binary_dataset_path=binary_path)
        ensure_binary_dataset(path, binary_path)
        return binary_path
    
    def _resolve_dataset_path(self, dataset_key: str) -> tuple[str, str, dict[int, str]]:
        """Resolve dataset path and metadata."""
        dataset_path, dataset_name, class_names = self._get_dataset_metadata(dataset_key)
        if dataset_key == "binary":
            dataset_path = self._prepare_binary_dataset()
        else:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Original dataset not found at {dataset_path}")
        return dataset_path, dataset_name, class_names
    
    def run(self) -> None:
        """Execute machine learning experiments for all specified datasets."""
        print("Machine Learning Experiments Runner")
        print(f"Datasets: {self.dataset_keys}")
        print(f"Classical trials: {self.classical_trials}")
        print(f"DNN trials: {self.dnn_trials}")
        print(f"Verbose: {self.verbose}")
        
        for dataset_key in self.dataset_keys:
            try:
                dataset_path, dataset_name, _ = self._resolve_dataset_path(dataset_key)
                
                print(f"\n{'='*60}")
                print(f"Starting experiments for {dataset_name}")
                print(f"{'='*60}")
                
                # Run experiments
                results_path = train_and_evaluate(
                    dataset_key=dataset_key,
                    dataset_path=dataset_path,
                    dataset_label=dataset_name,
                    is_binary=(dataset_key == "binary"),
                    classical_trials=self.classical_trials,
                    dnn_trials=self.dnn_trials,
                    verbose=self.verbose,
                )
                
                print(f"\n{dataset_name} metrics saved to {results_path}")
                
            except FileNotFoundError as exc:
                print(f"Error processing {dataset_key}: {exc}")
                continue


def create_machine_learning_runner(
    dataset_keys: list[str] | None = None,
    regenerate_binary: bool = False,
    classical_trials: int | None = None,
    dnn_trials: int | None = None,
    verbose: bool = True
) -> MachineLearningRunner:
    """Factory function to create a machine learning runner."""
    config: dict[str, Any] = {
        'dataset_keys': dataset_keys or ['five', 'binary'],
        'regenerate_binary': regenerate_binary,
        'classical_trials': classical_trials,
        'dnn_trials': dnn_trials,
        'verbose': verbose
    }
    return MachineLearningRunner(config)


if __name__ == "__main__":
    # Example usage
    runner = create_machine_learning_runner(
        dataset_keys=['binary'],
        classical_trials=50,
        dnn_trials=25,
        verbose=True
    )
    runner.execute()