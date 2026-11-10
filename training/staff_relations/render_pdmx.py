# flake8: noqa: T201

"""
Renders PDMX scores for the training data of the staff relation model: single parts,
ensembles, choirs and piano music in other layouts than Lieder. The scores of the PDMX
development set (validation.staff_structure_pdmx) and scores with the same title are left
out.
Usage:
    poetry run python -m training.staff_relations.render_pdmx
    poetry run python -m training.staff_relations.create_dataset \\
        --sources datasets/staff_relations_pdmx_scores --directory datasets/staff_relations_pdmx
"""

import argparse
from pathlib import Path

from validation.staff_structure_pdmx import development_titles, render, select_scores

default_directory = Path("datasets/staff_relations_pdmx_scores")
scores_per_parts = {"1": 400, "2": 1000, "3": 1000, "4": 1000, "5+": 1000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=default_directory)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--single-parts", type=int, help="Only single parts, as many as given (other seed)"
    )
    args = parser.parse_args()
    counts = scores_per_parts if args.single_parts is None else {"1": args.single_parts}
    scores = select_scores(
        args.seed,
        counts,
        complexity_limit=100,
        excluded_titles=frozenset(development_titles()),
    )
    scores = [score for score in scores if score.exists()]
    print(f"Selected {len(scores)} scores", flush=True)
    render(scores, args.directory)
    for score in scores:
        (args.directory / score.name).unlink(missing_ok=True)
    print(f"Rendered {len(list(args.directory.glob('*.svg')))} pages to {args.directory}")


if __name__ == "__main__":
    main()
