import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))


def test_full_pipeline_uses_changed_config_for_plots(tmp_path):
    import subprocess
    config=json.loads((ROOT/"configs/semester.json").read_text())
    config.update(tasks=["sine"], label_noise_stds=[0], n_train=24, n_validation=12,
                  n_test=16, data_seeds=[71,82], measurement_seeds=[93], shots_per_setting=[8,16])
    configuration=tmp_path/"custom.json"
    configuration.write_text(json.dumps(config))
    results=tmp_path/"custom_results"
    figures=tmp_path/"custom_figures"
    for command in [
        ["run_study.py","--config",str(configuration),"--output",str(results)],
        ["make_figures.py","--results",str(results),"--figures",str(figures)],
        ["deep_audit.py","--results",str(results)],
    ]:
        completed=subprocess.run([sys.executable,str(ROOT/"scripts"/command[0]),*command[1:]],capture_output=True,text=True)
        assert completed.returncode==0,completed.stdout+completed.stderr
    assert len(list(figures.glob("*.png")))==6
    summary=json.loads((results/"summary.json").read_text())
    assert all(row["n_data_seeds"]==2 for row in summary["summary"])
    assert "five" not in summary["uncertainty"]
    assert json.loads((results/"deep_audit.json").read_text())["independent_numpy_fits"]==12
