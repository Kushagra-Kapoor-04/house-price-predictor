import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Computes Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Computes Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    Computes R² (Coefficient of Determination) score.
    R² = 1 - (SS_res / SS_tot)
    """
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Avoid division by zero
    if ss_total == 0:
        return 0.0
        
    return 1.0 - (ss_residual / ss_total)

def evaluate_predictions(y_true, y_pred):
    """
    Computes and returns a dictionary of evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }
