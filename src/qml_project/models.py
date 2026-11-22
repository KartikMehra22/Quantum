"""Training-only scaling, unpenalized intercept, validation-only ridge tuning."""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


def mse(target, prediction):
    return float(np.mean((np.asarray(target) - np.asarray(prediction)) ** 2))


def fit_readouts(x_train, y_train, x_validation, y_validation, lambdas):
    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train)
    validation = scaler.transform(x_validation)
    ordinary = LinearRegression().fit(train, y_train)
    # sklearn minimizes summed squared error + alpha ||w||^2.
    # Our report uses mean squared error + lambda ||w||^2, so alpha=n*lambda.
    candidates = []
    for penalty in lambdas:
        candidate = Ridge(alpha=len(train) * penalty, solver="svd").fit(train, y_train)
        candidates.append((mse(y_validation, candidate.predict(validation)), penalty, candidate))
    _, penalty, ridge = min(candidates, key=lambda entry: (entry[0], entry[1]))
    return scaler, {"OLS": (ordinary, 0.0), "Ridge": (ridge, float(penalty))}

