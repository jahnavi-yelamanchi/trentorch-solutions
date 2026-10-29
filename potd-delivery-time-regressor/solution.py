import numpy as np


def delivery_time_regressor(
    x: np.ndarray, y: np.ndarray, queries: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    queries = np.asarray(queries, dtype=float)

    n = x.shape[0]

    # Step 1: add a column of ones for the intercept -> shape (n, d+1)
    x_aug = np.hstack([np.ones((n, 1)), x])

    # Step 2: minimum-norm least squares via SVD-based pseudoinverse
    w = np.linalg.pinv(x_aug) @ y

    # Step 3: split into intercept and coefficients
    intercept = float(w[0])
    coefficients = w[1:]

    # Step 4: predict: yhat = w0 + q . w[1:]
    predictions = intercept + queries @ coefficients

    return intercept, coefficients, predictions
