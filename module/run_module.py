import subprocess
from pathlib import Path

def run(input_files, output_files, params, **kwargs):
    """
    Optional Python wrapper used only if running the module outside the official
    OmniBenchmark entrypoint system.
    """

    matrix = input_files["data.matrix"]
    labels = input_files["data.true_labels"]

    pred_path = Path(output_files["analysis.prediction.cyannotool"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(Path(__file__).resolve().parents[1] / "cyanno_pipeline" / "run_cyanno.py"),
        str(matrix),
        str(labels),
        str(pred_path)
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
