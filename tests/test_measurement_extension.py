import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from measurement_extension.covariance import covariance_of_means,scale_covariance,SIGNS
from measurement_extension.models import penalized_readout,fit_methods,ridge_candidates

ROOT=Path(__file__).resolve().parents[1]


def test_ridge_search_contains_all_zero_gamma_options_at_matched_budget():
    config=json.loads((ROOT/"configs/measurement_extension.json").read_text())
    grid=ridge_candidates(config)
    assert len(grid)==32
    assert set(config["ridge_lambdas"])<=set(grid)


def test_covariance_from_counts_matches_expanded_shot_records():
    counts=np.array([[[3,1,2,2]]*3])
    estimate=covariance_of_means(counts)[0]
    expanded=np.repeat(SIGNS,counts[0,0],axis=0)
    expected=np.cov(expanded,rowvar=False,ddof=1)/len(expanded)
    for g in range(3):
        np.testing.assert_allclose(estimate[g*3:g*3+3,g*3:g*3+3],expected,atol=1e-15)
    np.testing.assert_array_equal(estimate[:3,3:],0)
    assert np.min(np.linalg.eigvalsh(estimate))>=-1e-14


@pytest.mark.parametrize("counts",[np.ones((2,3,4))*2,np.zeros((1,3,4),int),np.array([[[1,0,0,0]]*3]),np.ones((1,4),int),np.full((1,3,4),-1,int)])
def test_invalid_covariance_records_are_rejected(counts):
    with pytest.raises(ValueError):covariance_of_means(counts)


def test_unbiased_mean_covariance_agrees_with_known_sampling_distribution():
    p=np.array([0.1,0.2,0.3,0.4]); shots=32
    counts=np.random.default_rng(900).multinomial(shots,p,size=(4000,3))
    estimated=covariance_of_means(counts).mean(axis=0)
    mean=p@SIGNS
    expected=((SIGNS.T*p)@SIGNS-np.outer(mean,mean))/shots
    np.testing.assert_allclose(estimated[:3,:3],expected,atol=0.0008)


def test_covariance_propagates_through_feature_scaling():
    covariance=np.diag(np.arange(1,10.))
    covariance[0,1]=covariance[1,0]=0.2
    scales=np.linspace(0.4,2.0,9)
    expected=np.diag(1/scales)@covariance@np.diag(1/scales)
    np.testing.assert_allclose(scale_covariance(covariance,scales),expected)


def test_zero_gamma_generalized_solver_matches_ridge_with_free_intercept():
    rng=np.random.default_rng(333)
    x=rng.normal(size=(30,9))+2;y=rng.normal(size=30)+1.5
    for lam in [1e-6,0.01,10]:
        weights,intercept=penalized_readout(x,y,lam*np.eye(9))
        expected=Ridge(alpha=30*lam,solver="svd").fit(x,y)
        np.testing.assert_allclose(weights,expected.coef_,atol=1e-10)
        np.testing.assert_allclose(intercept,expected.intercept_,atol=1e-10)


def test_diagonal_ablation_and_full_model_agree_for_diagonal_noise():
    rng=np.random.default_rng(812)
    train=rng.normal(size=(24,9));val=rng.normal(size=(12,9))
    config=json.loads((ROOT/"configs/measurement_extension.json").read_text())
    _,cov,models=fit_methods(train,rng.normal(size=24),val,rng.normal(size=12),np.full((24,3,4),2,int),config)
    np.testing.assert_allclose(cov,np.diag(np.diag(cov)),atol=1e-15)
    np.testing.assert_allclose(models["FullCov"].weights,models["DiagonalCov"].weights,atol=1e-12)
    assert models["FullCov"].gamma==models["DiagonalCov"].gamma
    assert {models[m].candidates for m in ["Ridge","DiagonalCov","FullCov"]}=={32}


def test_deterministic_outcomes_have_zero_covariance_and_wrong_training_count_fails():
    counts=np.zeros((24,3,4),int);counts[:,:,0]=16
    np.testing.assert_array_equal(covariance_of_means(counts),0)
    config=json.loads((ROOT/"configs/measurement_extension.json").read_text())
    with pytest.raises(ValueError,match="training inputs"):
        fit_methods(np.ones((24,9)),np.ones(24),np.ones((12,9)),np.ones(12),counts[:12],config)


def test_exact_reference_uses_zero_covariance_and_gamma_zero():
    rng=np.random.default_rng(510)
    train=rng.normal(size=(24,9));validation=rng.normal(size=(12,9))+4
    config=json.loads((ROOT/"configs/measurement_extension.json").read_text())
    scaler,cov,models=fit_methods(train,rng.normal(size=24),validation,rng.normal(size=12),None,config)
    np.testing.assert_allclose(scaler.mean_,train.mean(axis=0))
    np.testing.assert_array_equal(cov,0)
    assert models["FullCov"].gamma==models["DiagonalCov"].gamma==0


def test_extension_rejects_original_seeds_and_mismatched_search_budget():
    import sys
    sys.path.insert(0,str(ROOT/"scripts"))
    from validate_extension import validate_extension
    baseline=json.loads((ROOT/"configs/semester.json").read_text())
    config=json.loads((ROOT/"configs/measurement_extension.json").read_text())
    validate_extension(config,baseline)
    config["data_seeds"][0]=11
    with pytest.raises(ValueError,match="disjoint"):validate_extension(config,baseline)
    config=json.loads((ROOT/"configs/measurement_extension.json").read_text())
    config["ridge_search"]["n_candidates"]=8
    with pytest.raises(ValueError,match="matched"):validate_extension(config,baseline)


def test_extension_pipeline_and_independent_audit(tmp_path):
    import subprocess
    import sys
    config=json.loads((ROOT/"configs/measurement_extension.json").read_text())
    config.update(tasks=["sine"],label_noise_stds=[0],n_train=24,n_validation=12,n_test=16,
                  data_seeds=[7001,7002],measurement_seeds=[7101],shots_per_setting=[8])
    path=tmp_path/"config.json";path.write_text(json.dumps(config))
    result=tmp_path/"run";figures=tmp_path/"figures";note=tmp_path/"results.md"
    commands=[
        ["run_extension.py","--config",str(path),"--output",str(result)],
        ["summarize_extension.py","--results",str(result),"--figures",str(figures),"--note",str(note)],
        ["audit_extension.py","--results",str(result),"--deep-physics"],
    ]
    for command in commands:
        run=subprocess.run([sys.executable,str(ROOT/"scripts"/command[0]),*command[1:]],capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
    verification=json.loads((result/"verification.json").read_text())
    assert verification["independent_model_evaluations"]==16
    assert verification["expanded_training_covariances_checked"]==2
    assert len(list(figures.glob("*.png")))==3
