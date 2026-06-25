import argparse
import numpy as np
import os

from src.data import generate_synthetic_data, load_california_data, train_test_split, standardize
from src.model import LinearRegression
from src.train import train_model
from src.evaluate import evaluate_predictions
from src.plot import generate_all_plots

def main():
    parser = argparse.ArgumentParser(description="Linear Regression House Price Predictor from Scratch")
    parser.add_argument("--dataset", type=str, choices=["california", "synthetic"], default="california",
                        help="Dataset to use: 'california' (default) or 'synthetic'")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--noise", type=float, default=1.0,
                        help="Standard deviation of Gaussian noise added to synthetic data (default: 1.0)")
    parser.add_argument("--n-features", type=int, default=5,
                        help="Number of features to generate for synthetic data (default: 5)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples to generate for synthetic data (default: 1000)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory to save the resulting plots (default: current directory)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("      LINEAR REGRESSION HOUSE PRICE PREDICTOR FROM SCRATCH      ")
    print("=" * 60)
    print(f"Dataset:       {args.dataset.upper()}")
    print(f"Epochs:        {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Test Split:    {args.test_size}")
    print(f"Random Seed:   {args.seed}")
    if args.dataset == "synthetic":
        print(f"Samples:       {args.n_samples}")
        print(f"Features:      {args.n_features}")
        print(f"Noise Std:     {args.noise}")
    print("-" * 60)
    
    # Set global seed for reproducibility
    np.random.seed(args.seed)
    
    # 1. Load / Generate Data
    if args.dataset == "california":
        print("Loading California Housing dataset...")
        X, y = load_california_data()
        feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
    else:
        print("Generating synthetic dataset...")
        X, y, true_w, true_b = generate_synthetic_data(
            n_samples=args.n_samples, 
            n_features=args.n_features, 
            noise=args.noise, 
            random_state=args.seed
        )
        feature_names = [f"Feature_{i+1}" for i in range(args.n_features)]
        
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Train split: {X_train.shape[0]} samples")
    print(f"Test split:  {X_test.shape[0]} samples")
    
    # 3. Standardization (Z-score Normalization)
    X_train_scaled, X_test_scaled = standardize(X_train, X_test)
    print("Features normalized successfully.")
    print("-" * 60)
    
    # 4. Initialize Model
    model = LinearRegression(n_features=X_train_scaled.shape[1], random_state=args.seed)
    
    # 5. Train Model
    print("Starting training...")
    loss_history, val_loss_history = train_model(
        model, 
        X_train_scaled, 
        y_train, 
        epochs=args.epochs, 
        learning_rate=args.lr, 
        X_val=X_test_scaled, 
        y_val=y_test, 
        verbose=True
    )
    print("Training finished.")
    print("-" * 60)
    
    # 6. Evaluate Model
    y_test_pred = model.forward(X_test_scaled)
    metrics = evaluate_predictions(y_test, y_test_pred)
    
    print("EVALUATION METRICS (TEST SET):")
    print(f"Mean Squared Error (MSE):   {metrics['MSE']:.6f}")
    print(f"Mean Absolute Error (MAE):  {metrics['MAE']:.6f}")
    print(f"R² Score (R-squared):       {metrics['R2']:.6f}")
    print("-" * 60)
    
    # Print model parameters
    print("LEARNED PARAMETERS:")
    print(f"Bias (Intercept): {model.bias[0] if isinstance(model.bias, np.ndarray) else model.bias:.6f}")
    for name, weight in zip(feature_names, model.weights.flatten()):
        print(f"Weight for {name:12}: {weight:.6f}")
    print("-" * 60)
    
    # 7. Plotting
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"{args.dataset}_")
    print("Generating performance visualizations...")
    generate_all_plots(loss_history, val_loss_history, y_test, y_test_pred, prefix=prefix)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
