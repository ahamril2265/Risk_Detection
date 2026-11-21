"""
Runs the Credit Risk Scoring notebook pipeline (01-15) in sequence, headlessly.

Each notebook is executed with nbclient and saved back in place (outputs and
execution counts included), exactly as if you had opened it in Jupyter and
chosen "Restart Kernel and Run All Cells". Execution stops at the first
notebook that raises an error, so a partial run never silently continues on
top of a broken state.

Usage (run from inside the notebooks/ directory, or anywhere -- paths are
resolved relative to this script's own location):

    python run_pipeline.py                  # run the full 01 -> 15 sequence
    python run_pipeline.py --from 6          # resume starting at notebook 06
    python run_pipeline.py --from 6 --to 12  # run only notebooks 06 through 12
    python run_pipeline.py --only 13 14 15   # run an explicit subset, in order
    python run_pipeline.py --timeout 1800    # override the 1200s per-cell timeout

A timestamped log is written to notebooks/pipeline_logs/ with a full record
of what ran, how long each notebook took, and the first error (if any).

Note on re-running notebooks 06-12: each of those notebooks' own "Save Model
Performance" cell unconditionally appends a row to results/model_results.csv
(and model_results_extended.csv) with no check for whether that model is
already present. Running the same notebook more than once against the same
results files -- e.g. resuming a pipeline after a mid-run failure -- would
silently leave duplicate rows behind, which then corrupts anything downstream
that assumes one row per model (Notebook 13's ranking and "second-best model"
selection, in particular). This script closes that gap itself, at the
orchestration level, by deduplicating those two files (keeping the most
recently written row for each model) after every notebook run, rather than
editing the notebooks' own cells.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK_DIR = Path(__file__).resolve().parent
LOG_DIR = NOTEBOOK_DIR / "pipeline_logs"
RESULTS_DIR = NOTEBOOK_DIR / ".." / "results"

# (filename, columns that identify a unique model/variant row) for every
# results CSV that is built by per-notebook unconditional appends rather than
# a single notebook overwriting the whole file in one shot.
ACCUMULATING_RESULT_FILES = [
    ("model_results.csv", ["Model"]),
    ("model_results_extended.csv", ["Model", "Variant"]),
]


def dedupe_accumulating_results():
    """Remove duplicate model rows left behind by re-running notebooks 06-12.
    Keeps the last (most recently written) row for each model/variant. A no-op,
    harmless to call even when there is nothing to clean up."""
    for filename, key_columns in ACCUMULATING_RESULT_FILES:
        path = (RESULTS_DIR / filename).resolve()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        before = len(df)
        df = df.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)
        after = len(df)
        if after != before:
            df.to_csv(path, index=False)
            print(f"Deduplicated {filename}: {before} -> {after} rows "
                  f"(removed {before - after} stale duplicate(s) from a prior run).")

PIPELINE = [
    "01_Data_Understanding.ipynb",
    "02_Data_Preprocessing.ipynb",
    "03_Exploratory_Data_Analysis.ipynb",
    "04_Hypothesis_Testing.ipynb",
    "05_Feature_Engineering_and_PCA.ipynb",
    "06_Logistic_Regression.ipynb",
    "07_Decision_Tree.ipynb",
    "08_KNN.ipynb",
    "09_Naive_Bayes.ipynb",
    "10_Random_Forest.ipynb",
    "11_XGBoost.ipynb",
    "12_LightGBM.ipynb",
    "13_Credit_Risk_Scoring.ipynb",
    "14_Fairness_Analysis.ipynb",
    "15_Journal_Publication_Package.ipynb",
]


def notebook_number(name: str) -> int:
    return int(name[:2])


class Tee:
    """Writes to both stdout and a log file at once, so progress is visible
    live and also captured for the record."""

    def __init__(self, log_path: Path):
        self._log = open(log_path, "a", encoding="utf-8")

    def write(self, text: str):
        sys.__stdout__.write(text)
        self._log.write(text)
        self._log.flush()

    def flush(self):
        sys.__stdout__.flush()
        self._log.flush()

    def close(self):
        self._log.close()


def run_one(nb_path: Path, timeout: int) -> bool:
    print(f"\n{'=' * 70}")
    print(f"START: {nb_path.name}  [{time.strftime('%H:%M:%S')}]")
    print("=" * 70)
    t0 = time.time()

    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    status = "OK"
    try:
        client.execute()
    except CellExecutionError as e:
        status = "FAILED"
        print(f"\n!!! CELL EXECUTION ERROR in {nb_path.name} !!!")
        print(str(e)[:4000])
    except Exception:
        status = "FAILED"
        print(f"\n!!! UNEXPECTED ERROR in {nb_path.name} !!!")
        traceback.print_exc()
    finally:
        # Always save whatever progress was made, success or failure, so the
        # partially-run notebook is inspectable afterward instead of reverting
        # to its pre-run state.
        nbformat.write(nb, str(nb_path))

    elapsed = time.time() - t0
    print(f"\n{'-' * 70}")
    print(f"{status}: {nb_path.name}  ({elapsed / 60:.1f} min)")
    print("-" * 70)
    return status == "OK"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_n", type=int, default=None,
                         help="First notebook number to run (e.g. 6 for 06_Logistic_Regression.ipynb).")
    parser.add_argument("--to", dest="to_n", type=int, default=None,
                         help="Last notebook number to run, inclusive.")
    parser.add_argument("--only", dest="only", type=int, nargs="+", default=None,
                         help="Run exactly these notebook numbers, in the order given, ignoring --from/--to.")
    parser.add_argument("--timeout", type=int, default=1200,
                         help="Per-cell execution timeout in seconds (default: 1200 = 20 min). "
                              "A cell running longer than this raises an error instead of hanging silently.")
    args = parser.parse_args()

    if args.only:
        wanted = set(args.only)
        notebooks = [n for n in PIPELINE if notebook_number(n) in wanted]
        # Preserve the order the user asked for, not the pipeline's natural order.
        notebooks = sorted(notebooks, key=lambda n: args.only.index(notebook_number(n)))
    else:
        notebooks = [
            n for n in PIPELINE
            if (args.from_n is None or notebook_number(n) >= args.from_n)
            and (args.to_n is None or notebook_number(n) <= args.to_n)
        ]

    if not notebooks:
        print("No notebooks matched the given --from/--to/--only filters. Nothing to do.")
        sys.exit(1)

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print(f"Pipeline run starting: {len(notebooks)} notebook(s)")
        print(f"Log file: {log_path}")
        for n in notebooks:
            print(f"  - {n}")

        overall_start = time.time()
        for name in notebooks:
            nb_path = NOTEBOOK_DIR / name
            if not nb_path.exists():
                print(f"\nSTOPPING: {name} not found at {nb_path}")
                sys.exit(1)
            ok = run_one(nb_path, args.timeout)
            dedupe_accumulating_results()
            if not ok:
                print(f"\nSTOPPING: {name} failed. Fix the error above, then re-run "
                      f"with --from {notebook_number(name)} to resume from here.")
                sys.exit(1)

        total = time.time() - overall_start
        print(f"\nALL {len(notebooks)} NOTEBOOKS COMPLETED SUCCESSFULLY in {total / 60:.1f} min")
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == "__main__":
    main()
