#!/usr/bin/env python3
import sys
from pathlib import Path
import gzip
import pandas as pd

# --- Make sure `cyanno_pipeline` can be imported even when run as a script ---

THIS_DIR = Path(__file__).resolve().parent        # .../cyanno_pipeline
REPO_ROOT = THIS_DIR.parent                       # .../<repo_root> that contains cyanno_pipeline/

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cyanno_pipeline.cyanno import CyAnnoClassifier


def main(matrix_path, labels_path, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load matrix
    with gzip.open(matrix_path, "rt") as f:
        matrix_df = pd.read_csv(f, sep=",")

    # Load true labels
    with gzip.open(labels_path, "rt") as f:
        labels_df = pd.read_csv(f, header=None, names=["cell_type"])

    if len(matrix_df) != len(labels_df):
        raise ValueError(
            f"Matrix and labels row count mismatch: "
            f"{len(matrix_df)} rows in matrix vs {len(labels_df)} in labels"
        )

    # Combine features + labels
    train_df = pd.concat([matrix_df, labels_df], axis=1).dropna(subset=["cell_type"])

    clf = CyAnnoClassifier(markers=list(matrix_df.columns))
    clf.train(train_df)

    preds, _ = clf.predict(matrix_df)

    # One label per line, no header
    pd.Series(preds).to_csv(output_file, index=False, header=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: run_cyanno.py <matrix.gz> <true_labels.gz> <output_file>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
