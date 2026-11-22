import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from validate_config import validate_config
from deep_audit import expected_keys

@pytest.mark.parametrize("field,value", [
    ("shots_per_setting", [32,32]), ("shots_per_setting", [32.5]),
    ("data_seeds", [11]), ("n_train", 0), ("n_validation", True),
    ("ridge_lambdas", [float("nan")]), ("label_noise_stds", [0,-0.1]),
    ("input_domain", [1, -1]), ("features", ["Z0"]), ("tasks", ["unknown"]),
])
def test_invalid_configurations_fail_explicitly(field, value):
    config = json.loads((ROOT / "configs/semester.json").read_text())
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_default_configuration_and_condition_coverage():
    config = json.loads((ROOT / "configs/semester.json").read_text())
    validate_config(config)
    assert len(expected_keys(config)) == 640


def test_partial_run_is_not_overwritten(tmp_path):
    import subprocess
    output = tmp_path / "partial"
    output.mkdir()
    sentinel = output / "config.json"
    sentinel.write_text("partial data must survive")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_study.py"), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "partial runs are preserved" in result.stderr
    assert sentinel.read_text() == "partial data must survive"


