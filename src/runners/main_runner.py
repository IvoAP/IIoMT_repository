from typing import Any

from . import BaseRunner
from .binary_dataset_runner import create_binary_dataset_runner
from .mutual_information_runner import create_mutual_information_runner
from .machine_learning_runner import create_machine_learning_runner


class MainRunner(BaseRunner):
    """Main orchestrator runner that can execute all project workflows."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.workflow = self.config.get('workflow', 'full')  # 'full', 'data', 'mi', 'ml'
        self.datasets = self.config.get('datasets', ['five', 'binary'])
        
        # Binary dataset configuration
        self.regenerate_binary = self.config.get('regenerate_binary', False)
        
        # Mutual information configuration
        self.mi_top_n = self.config.get('mi_top_n', 10)
        self.mi_alpha = self.config.get('mi_alpha', 2.0)
        
        # Machine learning configuration
        self.classical_trials = self.config.get('classical_trials', None)
        self.dnn_trials = self.config.get('dnn_trials', None)
        self.verbose = self.config.get('verbose', True)
    
    def run(self) -> None:
        """Execute the specified workflow."""
        print("IIoMT Project Main Runner")
        print(f"Workflow: {self.workflow}")
        print(f"Datasets: {self.datasets}")
        print("="*60)
        
        if self.workflow in ['full', 'data']:
            self._run_binary_dataset_workflow()
        
        if self.workflow in ['full', 'mi']:
            self._run_mutual_information_workflow()
        
        if self.workflow in ['full', 'ml']:
            self._run_machine_learning_workflow()
        
        print("\n" + "="*60)
        print("Main workflow completed successfully!")
    
    def _run_binary_dataset_workflow(self) -> None:
        """Run binary dataset generation workflow."""
        print("\n[1/3] Running Binary Dataset Generation...")
        binary_runner = create_binary_dataset_runner(
            force_regenerate=self.regenerate_binary
        )
        binary_runner.execute()
    
    def _run_mutual_information_workflow(self) -> None:
        """Run mutual information analysis workflow."""
        print("\n[2/3] Running Mutual Information Analysis...")
        mi_runner = create_mutual_information_runner(
            dataset_keys=self.datasets,
            top_n=self.mi_top_n,
            alpha=self.mi_alpha,
            regenerate_binary=False  # Already handled in data workflow
        )
        mi_runner.execute()
    
    def _run_machine_learning_workflow(self) -> None:
        """Run machine learning experiments workflow."""
        print("\n[3/3] Running Machine Learning Experiments...")
        ml_runner = create_machine_learning_runner(
            dataset_keys=self.datasets,
            regenerate_binary=False,  # Already handled in data workflow
            classical_trials=self.classical_trials,
            dnn_trials=self.dnn_trials,
            verbose=self.verbose
        )
        ml_runner.execute()


def create_main_runner(
    workflow: str = 'full',
    datasets: list[str] | None = None,
    regenerate_binary: bool = False,
    mi_top_n: int = 10,
    mi_alpha: float = 2.0,
    classical_trials: int | None = None,
    dnn_trials: int | None = None,
    verbose: bool = True
) -> MainRunner:
    """Factory function to create the main runner."""
    config: dict[str, Any] = {
        'workflow': workflow,
        'datasets': datasets or ['five', 'binary'],
        'regenerate_binary': regenerate_binary,
        'mi_top_n': mi_top_n,
        'mi_alpha': mi_alpha,
        'classical_trials': classical_trials,
        'dnn_trials': dnn_trials,
        'verbose': verbose
    }
    return MainRunner(config)


if __name__ == "__main__":
    # Example usage: Full workflow
    runner = create_main_runner(
        workflow='full',
        datasets=['binary'],
        regenerate_binary=True,
        mi_top_n=15,
        classical_trials=50,
        dnn_trials=25
    )
    runner.execute()