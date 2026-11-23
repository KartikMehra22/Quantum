"""Run the recorded extension; existing results are never overwritten."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from validate_extension import validate_extension
from measurement_extension.experiment import run_extension


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/measurement_extension.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/measurement_extension")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    validate_extension(config,json.loads((ROOT/"configs/semester.json").read_text()))
    run_extension(args.config,args.output,ROOT)
