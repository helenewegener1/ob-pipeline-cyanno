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


def main(train_matrix_path, train_labels_path, test_matrix_path, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Load training set ---
    with gzip.open(train_matrix_path, "rt") as f:
        train_matrix = pd.read_csv(f, sep=",")
    with gzip.open(train_labels_path, "rt") as f:
        train_labels = pd.read_csv(f, header=None, names=["cell_type"])

    # Drop rows with missing labels in training set
    valid_idx = train_labels["cell_type"].notna()
    train_matrix = train_matrix[valid_idx].reset_index(drop=True)
    train_labels = train_labels[valid_idx].reset_index(drop=True)

    # Combine features + labels for training
    train_df = pd.concat([train_matrix, train_labels], axis=1)

    # --- Train classifier ---
    clf = CyAnnoClassifier(markers=list(train_matrix.columns))
    clf.train(train_df)

    # --- Load test set ---
    with gzip.open(test_matrix_path, "rt") as f:
        test_matrix = pd.read_csv(f, sep=",")
    with gzip.open(test_labels_path, "rt") as f:
        test_labels = pd.read_csv(f, header=None, names=["cell_type"])

    # --- Predict on test set ---
    preds, _ = clf.predict(test_matrix)

    if len(preds) != len(test_labels):
        raise ValueError(
            f"Predicted labels rows ({len(preds)}) do not match test labels ({len(test_labels)})"
        )

    # --- Save predictions ---
    pd.Series(preds).to_csv(output_file, index=False, header=False)
    print(f"✅ Predictions saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: run_cyanno.py <train_matrix.gz> <train_labels.gz> <test_matrix.gz> <test_labels.gz> <output_file>")
        sys.exit(1)

    train_matrix_path = sys.argv[1]
    train_labels_path = sys.argv[2]
    test_matrix_path = sys.argv[3]
    test_labels_path = sys.argv[4]
    output_file = sys.argv[5]

    main(train_matrix_path, train_labels_path, test_matrix_path, output_file)
