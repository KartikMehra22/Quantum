"""Run from the repository root: python scripts/run_study.py"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qml_project.experiment import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/semester.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/semester")
    args = parser.parse_args()
    run(args.config, args.output)

