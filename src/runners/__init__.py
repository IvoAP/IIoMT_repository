"""
Base runner class and utilities for the IIoMT project.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseRunner(ABC):
    """Base class for all runners in the project."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    @abstractmethod
    def run(self) -> None:
        """Execute the runner's main functionality."""
        pass
    
    def setup(self) -> None:
        """Setup method called before run(). Override if needed."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup method called after run(). Override if needed."""
        pass
    
    def execute(self) -> None:
        """Execute the complete runner workflow."""
        try:
            self.setup()
            self.run()
        finally:
            self.cleanup()


# Import all runners for easy access
__all__ = ['BaseRunner']