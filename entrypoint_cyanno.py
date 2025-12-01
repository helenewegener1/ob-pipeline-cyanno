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
    parser.add_argument("--data.matrix", dest="matrix", type=str, required=True)
    parser.add_argument("--data.true_labels", dest="labels", type=str, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match your YAML: {dataset}_predicted_labels.txt
    output_file = output_dir / f"{args.name}_predicted_labels.txt"
    print(f"📄 Output will be saved to: {output_file}", flush=True)

    # IMPORTANT: resolve run_cyanno.py relative to this file
    repo_root = Path(__file__).resolve().parent
    run_script = repo_root / "cyanno_pipeline" / "run_cyanno.py"

    if not run_script.exists():
        raise FileNotFoundError(f"run_cyanno.py not found at {run_script}")

    # Use the same Python interpreter as the current process
    cmd = [
        sys.executable,
        str(run_script),
        args.matrix,
        args.labels,
        str(output_file),
    ]

    print("🚀 Running CyAnno pipeline:", " ".join(cmd), flush=True)

    # Let stdout/stderr flow through so Snakemake captures real errors
    result = subprocess.run(cmd)

    if result.returncode != 0:
        # propagate CyAnno’s exit code; its traceback will be in stderr.log
        sys.exit(result.returncode)

    print(f"🎉 SUCCESS — prediction saved to {output_file}", flush=True)


if __name__ == "__main__":
    main()
