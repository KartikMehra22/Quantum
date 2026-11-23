"""Extension-specific validation, including disjointness from the old study."""
import math
import numpy as np
from validate_config import validate_config


def validate_extension(config,baseline):
    validate_config(config)
    if config.get("methods")!=["OLS","Ridge","DiagonalCov","FullCov"]:
        raise ValueError("The four prespecified methods are required")
    if min(config["shots_per_setting"])<2:
        raise ValueError("Covariance estimation requires S >= 2")
    for field in ["data_seeds","measurement_seeds"]:
        if set(config[field])&set(baseline[field]):
            raise ValueError(f"Fresh {field} must be disjoint from the original study")
    gammas=config.get("covariance_gammas",[])
    if not gammas or 0 not in gammas or len(set(gammas))!=len(gammas) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in gammas):
        raise ValueError("Gamma grid must be finite, nonnegative, unique and include zero")
    search=config.get("ridge_search",{})
    if search.get("n_candidates")!=len(gammas)*len(config["ridge_lambdas"]):
        raise ValueError("Regularized methods must have matched candidate counts")
    bounds=[search.get("log10_min"),search.get("log10_max")]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in bounds) or bounds[0]>=bounds[1]:
        raise ValueError("Invalid logarithmic Ridge search bounds")
    points=search.get("logspace_points")
    if type(points) is not int or points<2:
        raise ValueError("At least two logarithmic search points are required")
    grid=np.unique(np.r_[np.logspace(*bounds,points),config["ridge_lambdas"]])
    if len(grid)!=search["n_candidates"]:
        raise ValueError("Combined Ridge grid must have exactly the matched number of candidates")

