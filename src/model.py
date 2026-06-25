import numpy as np

class LinearRegression:
    def __init__(self, n_features, random_state=None):
        """
        Initializes weights and bias.
        Weights are initialized randomly using standard normal (scaled by 0.01)
        Bias is initialized to zero.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0.0

    def forward(self, X):
        """
        Computes the forward pass: y_pred = X * w + b.
        X shape: (n_samples, n_features)
        Returns: predictions of shape (n_samples, 1)
        """
        return np.dot(X, self.weights) + self.bias

    def compute_loss(self, y_pred, y):
        """
        Computes Mean Squared Error (MSE) loss:
        L = (1/n) * sum((y_pred - y)^2)
        """
        return np.mean((y_pred - y) ** 2)

    def compute_gradients(self, X, y_pred, y):
        """
        Computes analytical gradients:
        dL/dw = (2/n) * X^T * (y_pred - y)
        dL/db = (2/n) * sum(y_pred - y)
        """
        n = X.shape[0]
        error = y_pred - y
        dw = (2.0 / n) * np.dot(X.T, error)
        db = (2.0 / n) * np.sum(error)
        return dw, db

    def update_parameters(self, dw, db, learning_rate):
        """
        Updates weights and bias using gradient descent:
        w = w - alpha * dw
        b = b - alpha * db
        """
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
