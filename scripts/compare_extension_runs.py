"""Compare fresh extension reproduction, including coefficients and predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from measurement_extension.analysis import read_rows


def compare(original,reproduction):
    original,reproduction=Path(original),Path(reproduction)
    if json.loads((original/"config.json").read_text())!=json.loads((reproduction/"config.json").read_text()):raise ValueError("Changed config")
    a,b=read_rows(original),read_rows(reproduction)
    if len(a)!=len(b):raise ValueError("Changed row count")
    maximum=0.
    for left,right in zip(a,b):
        if left.keys()!=right.keys():raise ValueError("Changed schema")
        for k,v in left.items():
            if isinstance(v,(float,int)):
                np.testing.assert_allclose(v,right[k],atol=1e-12,rtol=1e-10,err_msg=k);maximum=max(maximum,abs(v-right[k]))
            elif v!=right[k]:raise ValueError(f"Changed key or pointer: {k}")
    files={p.name for p in (original/"raw").glob("*.npz")}
    if files!={p.name for p in (reproduction/"raw").glob("*.npz")}:raise ValueError("Changed raw file set")
    for relative in ["fitted_models.npz",*["raw/"+p for p in sorted(files)]]:
        with np.load(original/relative) as x,np.load(reproduction/relative) as y:
            if set(x.files)!=set(y.files):raise ValueError("Changed array keys")
            for k in x.files:
                if np.issubdtype(x[k].dtype,np.integer):np.testing.assert_array_equal(x[k],y[k])
                else:np.testing.assert_allclose(x[k],y[k],rtol=1e-10,atol=1e-12)
    outcome=dict(status="passed",readouts_compared=len(a),raw_files_compared=len(files),fitted_model_arrays_compared=5,max_absolute_metric_difference=maximum)
    (original/"reproduction_verification.json").write_text(json.dumps(outcome,indent=2)+"\n");print(json.dumps(outcome,indent=2))
    return outcome


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("original",type=Path);p.add_argument("reproduction",type=Path)
    a=p.parse_args();compare(a.original,a.reproduction)
