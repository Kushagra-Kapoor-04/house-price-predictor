import matplotlib.pyplot as plt
import numpy as np

# Apply a clean, modern aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def plot_loss_curve(loss_history, val_loss_history=None, save_path="loss_curve.png"):
    """
    Plots training (and validation) loss vs epoch to verify convergence.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(loss_history) + 1)
    
    plt.plot(epochs, loss_history, label="Training Loss", color="#1f77b4", linewidth=2.5)
    if val_loss_history:
        plt.plot(epochs, val_loss_history, label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        
    plt.title("Model Convergence: Loss vs Epoch", fontweight="bold", pad=15)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (Loss)")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to: {save_path}")

def plot_predictions_vs_actual(y_true, y_pred, save_path="pred_vs_actual.png"):
    """
    Plots a scatter plot of predicted vs actual values alongside a 45-degree identity line.
    """
    plt.figure(figsize=(8, 6))
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Scatter plot
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, color="#2ca02c", edgecolors="none", s=25, label="Predictions")
    
    # Identity line (y = x)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Identity Line (Ideal)")
    
    plt.title("Model Performance: Predicted vs Actual", fontweight="bold", pad=15)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved predicted vs actual plot to: {save_path}")

def plot_residuals_histogram(y_true, y_pred, save_path="residuals_hist.png"):
    """
    Plots a histogram of the prediction residuals (actual - predicted).
    """
    plt.figure(figsize=(8, 5))
    
    residuals = y_true.flatten() - y_pred.flatten()
    
    # Histogram of residuals
    n, bins, patches = plt.hist(residuals, bins=30, density=True, alpha=0.75, color="#9467bd", edgecolor="white")
    
    # Add a normal distribution curve for reference
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    y_pdf = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.plot(bins, y_pdf, color="#17becf", linewidth=2, label=f"Normal Fit\n(μ={mu:.3f}, σ={sigma:.3f})")
    
    plt.title("Analysis of Residuals (Errors)", fontweight="bold", pad=15)
    plt.xlabel("Residual Value (Actual - Predicted)")
    plt.ylabel("Density")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved residuals histogram to: {save_path}")

def generate_all_plots(loss_history, val_loss_history, y_true, y_pred, prefix=""):
    """
    Generates and saves all three standard plots.
    """
    plot_loss_curve(loss_history, val_loss_history, save_path=f"{prefix}loss_curve.png")
    plot_predictions_vs_actual(y_true, y_pred, save_path=f"{prefix}pred_vs_actual.png")
    plot_residuals_histogram(y_true, y_pred, save_path=f"{prefix}residuals_hist.png")
