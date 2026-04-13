import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)

    n_samples, n_features = X.shape
    W = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, W) + b
        y_cap = _sigmoid(z)

        error = y_cap - y

        dw = (1/n_samples) * np.dot(X.T, error)
        db = (1/n_samples) * np.sum(error)

        W -= lr * dw
        b -= lr * db

    return W, b