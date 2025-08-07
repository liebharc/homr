import numpy as np

def softmax(z, dim=-1):
    """Computes softmax function with support for dimension parameter.
    
    Args:
        z: input array
        dim: dimension along which softmax will be computed (-1 for last dimension)
    
    Returns an array of outputs with the same shape as z.
    """
    # For numerical stability: subtract the maximum along the specified axis
    shiftz = z - np.max(z, axis=dim, keepdims=True)
    exps = np.exp(shiftz)
    return exps / np.sum(exps, axis=dim, keepdims=True)