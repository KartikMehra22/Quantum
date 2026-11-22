"""Reject unsupported or ambiguous study settings before writing any results."""
import math

FEATURES = ["X0", "X1", "X0X1", "Y0", "Y1", "Y0Y1", "Z0", "Z1", "Z0Z1"]


def validate_config(config):
    if config.get("qubits") != 2 or isinstance(config.get("qubits"), bool):
        raise ValueError("Only the specified two-qubit circuit is supported")
    if config.get("features") != FEATURES or config.get("measurement_settings") != ["XX", "YY", "ZZ"]:
        raise ValueError("Feature names/order and measurement settings must match the implemented circuit")
    for field, minimum in [("n_train", 2), ("n_validation", 1), ("n_test", 1)]:
        value = config.get(field)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    for field in ["tasks", "shots_per_setting", "data_seeds", "measurement_seeds", "ridge_lambdas", "label_noise_stds"]:
        values = config.get(field)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{field} must be a nonempty list with no duplicates")
    if not set(config["tasks"]) <= {"sine", "mixed_harmonics"}:
        raise ValueError("Unknown target function")
    for field, minimum in [("shots_per_setting", 1), ("data_seeds", 0), ("measurement_seeds", 0)]:
        if any(type(v) is not int or v < minimum for v in config[field]):
            raise ValueError(f"{field} requires integers >= {minimum}")
    if len(config["data_seeds"]) < 2:
        raise ValueError("At least two dataset seeds are required for sample SD")
    for field, strictly_positive in [("ridge_lambdas", True), ("label_noise_stds", False)]:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or v < 0 or (strictly_positive and v == 0) for v in config[field]):
            raise ValueError(f"{field} contains invalid values")
    if 0 not in config["label_noise_stds"]:
        raise ValueError("Include the clean-label primary condition (noise 0)")
    domain = config.get("input_domain")
    if not isinstance(domain, list) or len(domain) != 2 or not all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in domain) or domain[0] >= domain[1]:
        raise ValueError("input_domain requires two finite increasing bounds")

