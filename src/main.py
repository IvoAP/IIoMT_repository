"""Minimal main entry point: delega execução aos runners via argumentos simples."""
import argparse
from .runners.main_runner import create_main_runner


def main() -> None:
    parser = argparse.ArgumentParser(description="IIoMT Runner")
    parser.add_argument('--workflow', choices=['full', 'data', 'mi', 'ml'], default='full')
    parser.add_argument('--datasets', default='five,binary')
    parser.add_argument('--regenerate-binary', action='store_true')
    parser.add_argument('--mi-top-n', type=int, default=10)
    parser.add_argument('--mi-alpha', type=float, default=2.0)
    parser.add_argument('--classical-trials', type=int)
    parser.add_argument('--dnn-trials', type=int)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    runner = create_main_runner(
        workflow=args.workflow,
        datasets=datasets,
        regenerate_binary=args.regenerate_binary,
        mi_top_n=args.mi_top_n,
        mi_alpha=args.mi_alpha,
        classical_trials=args.classical_trials,
        dnn_trials=args.dnn_trials,
        verbose=not args.quiet,
    )
    runner.execute()


if __name__ == '__main__':
    main()
