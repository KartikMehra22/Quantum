"""Compare a fresh reproduction against the recorded experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
import numpy as np
from qml_project.analysis import read_metrics
from deep_audit import unique_index


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("original",type=Path)
    p.add_argument("reproduction",type=Path)
    args=p.parse_args()
    a,b=read_metrics(args.original/"metrics.csv"),read_metrics(args.reproduction/"metrics.csv")
    assert set(unique_index(a))==set(unique_index(b)),"Changed model/condition keys"
    assert json.loads((args.original/"config.json").read_text())==json.loads((args.reproduction/"config.json").read_text()),"Changed experiment configuration"
    assert len(a)==len(b)
    max_error=0.0
    for left,right in zip(a,b):
        assert left.keys()==right.keys()
        for key,value in left.items():
            if isinstance(value,(int,float)):
                np.testing.assert_allclose(value,right[key],rtol=1e-10,atol=1e-12,err_msg=key)
                max_error=max(max_error,abs(value-right[key]))
            else:
                assert value==right[key],key
    original_raw={file.name for file in (args.original/"raw").glob("*.npz")}
    reproduced_raw={file.name for file in (args.reproduction/"raw").glob("*.npz")}
    assert original_raw==reproduced_raw,"Missing or extra raw files"
    original_predictions=unique_index(json.loads((args.original/"predictions.json").read_text()))
    reproduced_predictions=unique_index(json.loads((args.reproduction/"predictions.json").read_text()))
    assert set(original_predictions)==set(reproduced_predictions),"Changed prediction keys"
    for identifier in original_predictions:
        np.testing.assert_allclose(original_predictions[identifier]["prediction"],reproduced_predictions[identifier]["prediction"],rtol=1e-10,atol=1e-12)
    for file in (args.original/"raw").glob("*.npz"):
        left,right=np.load(file),np.load(args.reproduction/"raw"/file.name)
        assert set(left.files)==set(right.files)
        for key in left.files:
            np.testing.assert_allclose(left[key],right[key],rtol=1e-12,atol=1e-14)
    result=dict(status="passed",rows_compared=len(a),raw_files_compared=len(original_raw),prediction_vectors_compared=len(original_predictions),max_absolute_metric_difference=max_error)
    (args.original/"reproduction_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
