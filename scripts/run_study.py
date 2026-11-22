"""Run from the repository root: python scripts/run_study.py"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qml_project.experiment import run
from validate_config import validate_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/semester.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/semester")
    args = parser.parse_args()
    validate_config(json.loads(args.config.read_text()))
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("Use an empty output directory; partial runs are preserved too")
    run(args.config, args.output)

