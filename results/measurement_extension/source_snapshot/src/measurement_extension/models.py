"""Four readouts with training-only covariance and matched candidate counts."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler
from .covariance import covariance_of_means,scale_covariance

METHODS=("OLS","Ridge","DiagonalCov","FullCov")


def ridge_candidates(config):
    search=config["ridge_search"]
    # Include every gamma=0 covariance candidate; the baseline cannot lose
    # merely because the alternative uses a different ridge-only grid.
    return np.unique(np.r_[np.logspace(search["log10_min"],search["log10_max"],search["logspace_points"]),config["ridge_lambdas"]])


@dataclass
class Readout:
    weights: np.ndarray
    intercept: float
    lambda_: float
    gamma: float
    validation_mse: float
    candidates: int

    def predict(self,standardized_features):
        return standardized_features @ self.weights + self.intercept


def penalized_readout(z,y,penalty):
    """Mean squared training loss + w^T penalty w; intercept unpenalized."""
    zmean,ymean=z.mean(axis=0),float(y.mean())
    centered=z-zmean
    weights=np.linalg.solve(centered.T@centered/len(z)+penalty,centered.T@(y-ymean)/len(z))
    return weights,float(ymean-zmean@weights)


def fit_methods(train,y,validation,y_validation,training_counts,config):
    if training_counts is not None and len(training_counts)!=len(train):
        raise ValueError("Covariance must be estimated from the training inputs only")
    scaler=StandardScaler().fit(train)
    z,v=scaler.transform(train),scaler.transform(validation)
    # Only training counts enter the estimated measurement-noise penalty.
    covariance=np.zeros((train.shape[1],train.shape[1])) if training_counts is None else covariance_of_means(training_counts).mean(axis=0)
    covariance=scale_covariance(covariance,scaler.scale_)
    models={}
    ordinary=LinearRegression().fit(z,y)
    models["OLS"]=Readout(ordinary.coef_.copy(),float(ordinary.intercept_),0.,0.,float(np.mean((ordinary.predict(v)-y_validation)**2)),1)
    ridge_grid=ridge_candidates(config)
    candidates=[]
    for lam in ridge_grid:
        model=Ridge(alpha=len(z)*lam,solver="svd").fit(z,y)
        loss=float(np.mean((model.predict(v)-y_validation)**2))
        candidates.append((loss,float(lam),model))
    loss,lam,model=min(candidates,key=lambda entry:(entry[0],entry[1]))
    models["Ridge"]=Readout(model.coef_.copy(),float(model.intercept_),lam,0.,loss,len(candidates))
    for name,c in [("DiagonalCov",np.diag(np.diag(covariance))),("FullCov",covariance)]:
        candidates=[]
        for lam in config["ridge_lambdas"]:
            for gamma in config["covariance_gammas"]:
                weights,intercept=penalized_readout(z,y,lam*np.eye(z.shape[1])+gamma*c)
                loss=float(np.mean((v@weights+intercept-y_validation)**2))
                candidates.append((loss,gamma,lam,weights,intercept))
        loss,gamma,lam,weights,intercept=min(candidates,key=lambda entry:(entry[0],entry[1],entry[2]))
        models[name]=Readout(weights,intercept,float(lam),float(gamma),loss,len(candidates))
    return scaler,covariance,models
