"""Load a recorded readout and predict without retuning or retraining."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from qml_project.quantum import measurement_probabilities,exact_features,sample_records
from .analysis import read_rows


@dataclass(frozen=True)
class SavedReadout:
    weights: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    record: dict
    domain: tuple

    def predict_features(self,features):
        features=np.asarray(features,dtype=float)
        if features.ndim==1:features=features[None,:]
        if features.ndim!=2 or features.shape[1]!=len(self.weights) or not np.isfinite(features).all():
            raise ValueError("Expected finite feature vectors with nine columns")
        return ((features-self.mean)/self.scale)@self.weights+self.intercept

    def predict_inputs(self,inputs,sampling_seed=9001):
        inputs=np.atleast_1d(np.asarray(inputs,dtype=float))
        if inputs.ndim!=1 or not inputs.size or not np.isfinite(inputs).all():
            raise ValueError("Inputs must be a nonempty finite vector")
        if np.any(inputs<self.domain[0]) or np.any(inputs>self.domain[1]):
            raise ValueError("Input is outside the evaluated training domain; extrapolation is not supported")
        if type(sampling_seed) is not int or sampling_seed<0:raise ValueError("Sampling seed must be a nonnegative integer")
        probabilities=measurement_probabilities(inputs)
        shots=self.record["shots"]
        if shots:
            counts,features=sample_records(probabilities,shots,np.random.default_rng(sampling_seed))
        else:
            counts,features=None,exact_features(probabilities)
        return dict(inputs=inputs,predictions=self.predict_features(features),features=features,counts=counts,
                    simulated_shots=len(inputs)*3*shots,sampling_seed=sampling_seed)


def load_readout(directory,method="FullCov",task="sine",shots=32,label_noise_std=0.0,data_seed=None,measurement_seed=None):
    directory=Path(directory)
    config=json.loads((directory/"config.json").read_text())
    ds=config["data_seeds"][0] if data_seed is None else data_seed
    ms=(-1 if shots==0 else config["measurement_seeds"][0]) if measurement_seed is None else measurement_seed
    matches=[r for r in read_rows(directory) if (r["method"],r["task"],r["shots"],r["label_noise_std"],r["data_seed"],r["measurement_seed"])==(method,task,shots,label_noise_std,ds,ms)]
    if len(matches)!=1:raise ValueError("Choose exactly one recorded method/task/budget/dataset/acquisition combination")
    record=matches[0];i=record["row_id"]
    with np.load(directory/"fitted_models.npz",allow_pickle=False) as models:
        weights,mean,scale=[models[k][i].copy() for k in ["weights","means","scales"]]
        intercept=float(models["intercepts"][i])
    if weights.shape!=(9,) or mean.shape!=(9,) or scale.shape!=(9,) or not all(np.isfinite(a).all() for a in [weights,mean,scale]) or not np.isfinite(intercept) or np.any(scale<=0):
        raise ValueError("Invalid recorded model parameters")
    for array in [weights,mean,scale]:array.setflags(write=False)
    return SavedReadout(weights,intercept,mean,scale,record.copy(),tuple(config["input_domain"]))
