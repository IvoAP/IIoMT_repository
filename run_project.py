#!/usr/bin/env python3
"""
IIoMT Project Runner Script

Usage examples:
    python run_project.py                                  # Full workflow
    python run_project.py --workflow data                  # Only binary dataset generation
    python run_project.py --workflow mi                    # Only mutual information
    python run_project.py --workflow ml                    # Only machine learning
    python run_project.py --datasets binary               # Only binary dataset
    python run_project.py --datasets five,binary          # Both datasets
    python run_project.py --regenerate-binary             # Force regenerate binary dataset
    python run_project.py --classical-trials 100          # Custom trials for classical models
    python run_project.py --dnn-trials 50                 # Custom trials for DNN
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from runners.main_runner import create_main_runner


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IIoMT Project Runner - Execute machine learning workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                     # Run full workflow on both datasets
  %(prog)s --workflow data                     # Only generate binary dataset
  %(prog)s --workflow mi --datasets binary    # Mutual information on binary dataset only
  %(prog)s --workflow ml --classical-trials 100  # ML experiments with 100 classical trials
  %(prog)s --regenerate-binary --verbose      # Force binary dataset regeneration with verbose output
        """
    )
    
    parser.add_argument(
        '--workflow',
        choices=['full', 'data', 'mi', 'ml'],
        default='full',
        help='Workflow to execute (default: full)'
    )
    
    parser.add_argument(
        '--datasets',
        default='five,binary',
        help='Comma-separated list of datasets: five,binary (default: five,binary)'
    )
    
    parser.add_argument(
        '--regenerate-binary',
        action='store_true',
        help='Force regeneration of binary dataset'
    )
    
    parser.add_argument(
        '--mi-top-n',
        type=int,
        default=10,
        help='Number of top features to display for mutual information (default: 10)'
    )
    
    parser.add_argument(
        '--mi-alpha',
        type=float,
        default=2.0,
        help='Alpha parameter for mutual information weight calculation (default: 2.0)'
    )
    
    parser.add_argument(
        '--classical-trials',
        type=int,
        help='Number of Optuna trials for classical models (default: from training module)'
    )
    
    parser.add_argument(
        '--dnn-trials',
        type=int,
        help='Number of Optuna trials for DNN model (default: from training module)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output during training'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Parse datasets
    datasets = [ds.strip() for ds in args.datasets.split(',') if ds.strip()]
    
    # Validate datasets
    valid_datasets = {'five', 'binary'}
    for dataset in datasets:
        if dataset not in valid_datasets:
            print(f"Error: Invalid dataset '{dataset}'. Valid options: {', '.join(valid_datasets)}")
            sys.exit(1)
    
    # Create and run the main runner
    try:
        runner = create_main_runner(
            workflow=args.workflow,
            datasets=datasets,
            regenerate_binary=args.regenerate_binary,
            mi_top_n=args.mi_top_n,
            mi_alpha=args.mi_alpha,
            classical_trials=args.classical_trials,
            dnn_trials=args.dnn_trials,
            verbose=not args.quiet
        )
        
        runner.execute()
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()