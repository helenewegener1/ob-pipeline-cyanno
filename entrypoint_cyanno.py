#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="CyAnno OmniBenchmark module")

    # Required OmniBenchmark arguments
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where outputs must be written."
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Dataset name used in output filename."
    )

    # Inputs defined in YAML
    parser.add_argument("--train.data.matrix", type=str, required=True, dest="train_data_matrix")
    parser.add_argument("--labels_train", type=str, required=True)
    parser.add_argument("--test.data.matrix", type=str, required=True, dest="test_data_matrix")
    parser.add_argument("--labels_test", type=str, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file (OmniBenchmark convention)
    output_file = output_dir / f"{args.name}_predicted_labels.txt"
    print(f"📄 Output will be saved to: {output_file}", flush=True)

    # Repo root in the cloned module (same level as `cyanno_pipeline/`)
    repo_root = Path(__file__).resolve().parent
    run_script = repo_root / "cyanno_pipeline" / "run_cyanno.py"

    # Build command to pass all four inputs
    cmd = [
        sys.executable,
        str(run_script),
        args.train_data_matrix,
        args.labels_train,
        args.test_data_matrix,
        args.labels_test,
        str(output_file),
    ]

    print("🚀 Running CyAnno pipeline:")
    print("   ", " ".join(cmd), flush=True)
    print(f"   (cwd = {repo_root})", flush=True)

    # Run the pipeline
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"❌ CyAnno crashed (exit {result.returncode})")

    print(f"🎉 SUCCESS — prediction saved to {output_file}", flush=True)


if __name__ == "__main__":
    main()
