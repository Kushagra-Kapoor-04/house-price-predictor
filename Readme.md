# House Price Predictor — Linear Regression from Scratch

A pure NumPy implementation of linear regression that predicts house prices using gradient descent — no scikit-learn, no ML libraries. Built to demonstrate a full understanding of the forward pass, loss computation, backpropagation, and weight updates at the mathematical level.

---

## What this project does

Most ML tutorials hand you `model.fit()` and call it a day. This project builds every step from scratch:

- **Forward pass** — computes predictions as `ŷ = Xw + b`
- **Loss function** — mean squared error computed manually
- **Gradient computation** — analytical gradients for weights and bias
- **Gradient descent** — iterative weight updates with configurable learning rate
- **Evaluation** — MSE, MAE, and R² computed without external libraries

Supports two datasets out of the box: the real-world **California Housing** dataset and a configurable **synthetic dataset** for controlled experiments.

---

## Project structure

```
house-price-predictor/
│
├── src/
│   ├── data.py          # Data loading, train-test split, standardisation
│   ├── model.py         # LinearRegression class (forward pass, weights, bias)
│   ├── train.py         # Training loop with validation tracking
│   ├── evaluate.py      # MSE, MAE, R² metrics
│   └── plot.py          # Loss curves, prediction vs actual plots
│
├── outputs/             # Generated plots saved here
├── main.py              # CLI entry point with argparse
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/Kushagra-Kapoor-04/house-price-predictor.git
cd house-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run on California Housing dataset (default)
python main.py

# Run on synthetic data with custom settings
python main.py --dataset synthetic --epochs 2000 --lr 0.005 --n-samples 1500
```

---

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `california` | `california` or `synthetic` |
| `--epochs` | `1000` | Number of training iterations |
| `--lr` | `0.01` | Learning rate for gradient descent |
| `--test-size` | `0.2` | Fraction of data held out for testing |
| `--seed` | `42` | Random seed for reproducibility |
| `--noise` | `1.0` | Gaussian noise std (synthetic only) |
| `--n-features` | `5` | Feature count (synthetic only) |
| `--n-samples` | `1000` | Sample count (synthetic only) |
| `--out-dir` | `.` | Directory to save output plots |

---

## Results

Running on the California Housing dataset with default settings:

| Metric | Score |
|---|---|
| R² (R-squared) | ~0.60 |
| MAE | ~0.53 |
| MSE | ~0.55 |

> Linear regression has a natural performance ceiling on this dataset due to non-linear relationships between features. The goal here is correctness of the implementation, not state-of-the-art accuracy.

---

## How gradient descent works here

At each epoch, the model computes:

```
Loss     = (1/n) * Σ (ŷᵢ - yᵢ)²
∂L/∂w   = (2/n) * Xᵀ(ŷ - y)
∂L/∂b   = (2/n) * Σ(ŷ - y)

w = w - lr * ∂L/∂w
b = b - lr * ∂L/∂b
```

No autograd. No PyTorch. Just math and NumPy.

---

## Sample output

```
============================================================
 LINEAR REGRESSION HOUSE PRICE PREDICTOR FROM SCRATCH
============================================================
Dataset: CALIFORNIA
Epochs: 1000 | LR: 0.01 | Test Split: 0.2
------------------------------------------------------------
Loaded 20640 samples with 8 features.
Train: 16512 samples | Test: 4128 samples
Features normalized successfully.
------------------------------------------------------------
Starting training...
Epoch  100/1000 | Train Loss: 0.8342 | Val Loss: 0.8501
Epoch  500/1000 | Train Loss: 0.5814 | Val Loss: 0.5923
Epoch 1000/1000 | Train Loss: 0.5501 | Val Loss: 0.5612
------------------------------------------------------------
EVALUATION METRICS (TEST SET):
MSE: 0.561204 | MAE: 0.532018 | R²: 0.601837
------------------------------------------------------------
```

---

## Key concepts demonstrated

- Linear algebra fundamentals (matrix multiplication, vector operations)
- Feature standardisation (Z-score normalisation) and why it matters for gradient descent convergence
- Train/test splitting without data leakage
- Tracking both training and validation loss to detect overfitting
- Learned weight interpretation (which features matter most)

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)

---

## What's next

This project is part of a larger AI/ML learning roadmap. Next steps building on this foundation:

- [ ] Add momentum and Adam optimiser variants
- [ ] Extend to polynomial regression (non-linear features)
- [ ] Regularisation: L1 (Lasso) and L2 (Ridge) from scratch
- [ ] Compare against scikit-learn's implementation on the same data

---

## Author

**Kushagra Kapoor**
[GitHub](https://github.com/Kushagra-Kapoor-04) · [LinkedIn](https://linkedin.com/in/YOUR-LINKEDIN-HANDLE)