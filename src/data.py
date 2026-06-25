import numpy as np

def generate_synthetic_data(n_samples=1000, n_features=5, noise=1.0, random_state=None):
    """
    Generates synthetic linear regression data.
    y = X * w + b + noise * epsilon
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random features from standard normal distribution
    X = np.random.randn(n_samples, n_features)
    
    # Generate true weights and bias
    true_w = np.random.uniform(-10, 10, size=(n_features, 1))
    true_b = np.random.uniform(-5, 5)
    
    # Compute targets with Gaussian noise
    epsilon = np.random.randn(n_samples, 1)
    y = np.dot(X, true_w) + true_b + noise * epsilon
    
    return X, y, true_w, true_b

def load_california_data():
    """
    Loads California housing dataset using sklearn (the only allowed sklearn import).
    Returns X (features) and y (targets) as numpy arrays.
    """
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1) # Reshape to 2D column vector (n_samples, 1)
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits the datasets into training and testing sets.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def standardize(X_train, X_test):
    """
    Standardizes features using Z-score normalization: (X - mean) / std.
    Computes mean and std on X_train to prevent data leakage.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Prevent division by zero for constant features
    std[std == 0.0] = 1.0
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled
