"""Run all technical completion checks and write a machine-readable receipt."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def verify(report):
    started=time.perf_counter()
    report=Path(report);report.parent.mkdir(parents=True,exist_ok=True)
    steps=[]
    # Temporary reproduction destinations preserve the canonical results and
    # disappear after the comparisons, avoiding stale output reuse.
    with tempfile.TemporaryDirectory(prefix="quantum-completion-") as workspace:
        workspace=Path(workspace)
        jobs=[
            ("automated_tests",["-m","pytest","-q","--junitxml",str(workspace/"tests.xml")]),
            ("original_independent_audit",["scripts/deep_audit.py","--deep-physics"]),
            ("extension_independent_audit",["scripts/audit_extension.py","--deep-physics"]),
            ("original_fresh_run",["scripts/run_study.py","--output",str(workspace/"baseline")]),
            ("original_reproduction",["scripts/compare_runs.py","results/semester",str(workspace/"baseline")]),
            ("extension_fresh_run",["scripts/run_extension.py","--output",str(workspace/"extension")]),
            ("extension_reproduction",["scripts/compare_extension_runs.py","results/measurement_extension",str(workspace/"extension")]),
            ("four_method_refit_demo",["scripts/defense_demo.py","--refit"]),
            ("new_input_prediction",["scripts/predict.py","--method","FullCov","--shots","32","--x","0.2","0.5"]),
        ]
        status="passed"
        for name,arguments in jobs:
            print(f"\n=== {name} ===",flush=True)
            tick=time.perf_counter()
            completed=subprocess.run([sys.executable,*arguments],cwd=ROOT)
            steps.append(dict(name=name,status="passed" if completed.returncode==0 else "failed",return_code=completed.returncode,duration_seconds=time.perf_counter()-tick))
            if completed.returncode:
                status="failed"
                break
        test_counts={}
        if (workspace/"tests.xml").exists():
            import xml.etree.ElementTree as ET
            suites=list(ET.parse(workspace/"tests.xml").getroot().iter("testsuite"))
            test_counts={field:sum(int(s.get(field,0)) for s in suites) for field in ["tests","failures","errors","skipped"]}
    try:
        revision=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    except (subprocess.CalledProcessError,FileNotFoundError):revision=None
    receipt=dict(status=status,completed_utc=datetime.now(timezone.utc).isoformat(),source_commit=revision,
                 python=sys.version.split()[0],duration_seconds=time.perf_counter()-started,
                 automated_tests=test_counts,steps=steps,
                 scope="Technical implementation, saved-model inference, original and extension audits, and fresh numerical reproduction. Does not claim a completed paper or general accuracy superiority.")
    report.write_text(json.dumps(receipt,indent=2)+"\n")
    print(f"\nTechnical verification {status}: {report}",flush=True)
    return 0 if status=="passed" else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,default=ROOT/"results/project_verification.json")
    raise SystemExit(verify(p.parse_args().report))
