import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N, D = X.shape
    w = np.zeros((D,), dtype=np.float64)
    b = 0.0
    for epoch in range(steps):
        logit = X@w+b
        out = _sigmoid(logit)
        error = out - y
        dw = (lr/N) * X.T @ error
        db = (lr/1.0*N) * np.sum(error)
        w -= dw
        b -= db

    return (w, b)
    