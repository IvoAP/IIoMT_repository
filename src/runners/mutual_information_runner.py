import os
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import BaseRunner
from ..constraints import BINARY_CLASS_NAMES, ORIGINAL_CLASS_NAMES
from ..file_utils import (
    binary_path, 
    ensure_binary_dataset, 
    generate_binary_dataset,
    load_and_split_data, 
    path
)


class MutualInformationRunner(BaseRunner):
    """Runner for computing mutual information scores."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.dataset_keys = self.config.get('dataset_keys', ['five', 'binary'])
        self.top_n = self.config.get('top_n', 10)
        self.alpha = self.config.get('alpha', 2.0)
        self.regenerate_binary = self.config.get('regenerate_binary', False)
        self.random_state = self.config.get('random_state', 42)
        
        # Ensure MI results directory
        self.mi_dir = self.results_dir / "mi"
        self.mi_dir.mkdir(exist_ok=True)
    
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
    
    def _compute_mutual_information(
        self, 
        dataset_key: str, 
        dataset_name: str, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        """Compute and save mutual information scores."""
        print(f"\nComputing mutual information scores for {dataset_name} ({dataset_key}) dataset:")
        
        # Process categorical columns
        X_processed = X.copy()
        categorical_cols = X_processed.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            X_processed[col] = X_processed[col].astype("category").cat.codes
        
        # Calculate mutual information scores
        scores = mutual_info_classif(X_processed.values, y.values, random_state=self.random_state)
        mi_df = pd.DataFrame({"feature": X.columns, "mutual_information": scores})
        mi_df.sort_values(by="mutual_information", ascending=False, inplace=True)
        
        # Calculate weights
        weights = mi_df["mutual_information"].clip(lower=0).pow(self.alpha)
        total_weight = weights.sum()
        if total_weight > 0:
            mi_df["weight_alpha"] = weights / total_weight
        else:
            mi_df["weight_alpha"] = 0.0
        mi_df["alpha"] = self.alpha
        
        # Display top features
        display_count = min(self.top_n, len(mi_df))
        for _, row in mi_df.head(display_count).iterrows():
            print(
                f"  {row['feature']}: MI={row['mutual_information']:.6f} | "
                f"weight(alpha={self.alpha:g})={row['weight_alpha']:.6f}"
            )
        
        # Save results
        output_path = self.mi_dir / f"{dataset_key}_mutual_information.csv"
        mi_df.to_csv(output_path, index=False)
        print(f"Full mutual information scores saved to {output_path}")
    
    def run(self) -> None:
        """Execute mutual information computation for all specified datasets."""
        print("Mutual Information Runner")
        print(f"Datasets: {self.dataset_keys}")
        print(f"Top features to display: {self.top_n}")
        print(f"Alpha parameter: {self.alpha}")
        
        for dataset_key in self.dataset_keys:
            try:
                dataset_path, dataset_name, _ = self._resolve_dataset_path(dataset_key)
                
                # Load data
                X, y = load_and_split_data(dataset_path, target_column="y")
                
                # Compute MI
                self._compute_mutual_information(dataset_key, dataset_name, X, y)
                
            except FileNotFoundError as exc:
                print(f"Error processing {dataset_key}: {exc}")
                continue


def create_mutual_information_runner(
    dataset_keys: list[str] | None = None,
    top_n: int = 10,
    alpha: float = 2.0,
    regenerate_binary: bool = False,
    random_state: int = 42
) -> MutualInformationRunner:
    """Factory function to create a mutual information runner."""
    config: dict[str, Any] = {
        'dataset_keys': dataset_keys or ['five', 'binary'],
        'top_n': top_n,
        'alpha': alpha,
        'regenerate_binary': regenerate_binary,
        'random_state': random_state
    }
    return MutualInformationRunner(config)


if __name__ == "__main__":
    # Example usage
    runner = create_mutual_information_runner(
        dataset_keys=['binary'],
        top_n=15,
        alpha=2.0
    )
    runner.execute()