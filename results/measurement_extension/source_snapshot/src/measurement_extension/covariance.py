"""Unbiased covariance of feature MEANS from shared joint outcome counts."""
import numpy as np

# Independent declaration for the outcome -> (single0, single1, pair) map.
SIGNS = np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])


def covariance_of_means(counts):
    """Shape (inputs, 9, 9); independent settings give block-diagonal noise.

    For S shots and single-shot feature vectors v, the unbiased covariance
    of the sample mean is (sum(v v^T)-S*m*m^T)/(S*(S-1)).
    This is within-input measurement uncertainty, NOT across-input covariance.
    """
    counts=np.asarray(counts)
    if counts.ndim!=3 or counts.shape[1:]!=(3,4):
        raise ValueError("Expected (inputs, 3 settings, 4 joint outcomes)")
    if not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0):
        raise ValueError("Counts must be nonnegative integers")
    totals=counts.sum(axis=-1)
    if np.any(totals<2):
        raise ValueError("At least two shots per setting are needed for covariance")
    output=np.zeros((len(counts),9,9))
    for i in range(len(counts)):
        for g in range(3):
            shots=int(totals[i,g])
            mean=counts[i,g] @ SIGNS / shots
            second=(SIGNS.T*counts[i,g]) @ SIGNS
            block=(second-shots*np.outer(mean,mean))/(shots*(shots-1))
            block=(block+block.T)/2
            values,vectors=np.linalg.eigh(block)
            if values.min() < -1e-12:
                raise ValueError("Covariance unexpectedly has a negative eigenvalue")
            block=(vectors*np.maximum(values,0)) @ vectors.T
            output[i,3*g:3*g+3,3*g:3*g+3]=block
    return output


def scale_covariance(covariance, scales):
    """Propagate noise through the same train-fitted feature standardization."""
    covariance,scales=np.asarray(covariance),np.asarray(scales)
    if covariance.shape!=(len(scales),len(scales)) or np.any(scales<=0) or not np.isfinite(covariance).all() or not np.isfinite(scales).all():
        raise ValueError("Invalid covariance or scale")
    return covariance/scales[:,None]/scales[None,:]

