import numpy as np
import matplotlib.pyplot as plt


def generate_polynomial_data(num_samples, coefficients, x_min=-10, x_max=10, noise_std=50, verbose=False):
    """
    Generate a dataset of 1D polynomial data points with optional noise.

    Parameters:
    - num_samples: Number of data points to generate.
    - coefficients: List of coefficients for the polynomial.
    - x_min, x_max: The range of x values.
    - noise_std: Standard deviation of Gaussian noise added to the outputs.
    - verbose: If True, print information about the generated data.

    Returns:
    - X: A NumPy matrix of shape (num_samples, len(coefficients)) where each
         column i represents the term x^i.
    - y: A NumPy array of outputs with added noise.
    """
    # Randomly choose 'num_samples' values of x within the range [x_min, x_max]
    x = np.random.uniform(x_min, x_max, (num_samples, 1))
    if verbose:
        print("First 5 x values:\n", x[:5])

    # Form the design matrix X, each column being x raised to a power
    X = np.vander(x.flatten(), len(coefficients), increasing=True)
    if verbose:
        print("First 5 rows of the design matrix X:\n", X[:5])

    # Calculate the hypothetical outputs without noise
    y_hat = X @ np.array(coefficients).reshape(-1, 1)
    if verbose:
        print("First 5 hypothetical outputs y_hat:\n", y_hat[:5])

    # Add Gaussian noise to the outputs
    y = y_hat + np.random.normal(0, noise_std, (num_samples, 1))
    if verbose:
        print("First 5 noisy outputs y:\n", y[:5])

    return X, y


def plot_data_predictions(x, y, y_hat, label_true='True Outputs', label_pred='Predicted Outputs', color_true='b',
                          color_pred='r'):
    """
    Plot the actual vs predicted data points.

    Parameters:
    - x: Vector of x coordinates.
    - y: Vector of true outputs.
    - y_hat: Vector of predicted outputs.
    - label_true: Label for the actual data plot.
    - color_true: Color for the actual data points.
    - label_pred: Label for the predicted data plot.
    - color_pred: Color for the predicted data points.
    """
    plt.plot(x, y, 'o', label=label_true, color=color_true)
    plt.plot(x, y_hat, '*', label=label_pred, color=color_pred)
    plt.grid()
    plt.legend()
    plt.show()


def initialize_weights(num_weights, zero_init=True):
    """
    Initialize the weights for the polynomial regression model.

    Parameters:
    - num_weights: The number of weights to generate.
    - zero_init: If True, initialize weights to zero; otherwise, use random values.

    Returns:
    - A NumPy array of weights.
    """
    if zero_init:
        weights = np.zeros((num_weights, 1))
    else:
        weights = np.random.uniform(-0.5, 0.5, (num_weights, 1))
    return weights


def perform_gradient_descent(X, y, initial_weights, learning_rate=0.00001, convergence_threshold=0.00001,
                             max_iterations=100, verbose=True):
    """
    Perform gradient descent to fit a polynomial regression model.

    Parameters:
    - X: Input features matrix.
    - y: Output values vector.
    - initial_weights: Initial weights vector.
    - learning_rate: Learning rate for the gradient descent.
    - convergence_threshold: Convergence threshold for stopping criteria.
    - max_iterations: Maximum number of iterations.
    - verbose: If True, print error information every 100 iterations.

    Returns:
    - weights: The optimized weights.
    - error_history: List of error values at each iteration.
    - iterations: The number of iterations performed.
    """
    weights = initial_weights
    error_history = []
    m = len(y)

    for iteration in range(max_iterations):
        predictions = X @ weights
        error = predictions - y
        mean_squared_error = np.mean(error ** 2)
        error_history.append(mean_squared_error)

        gradient = -2 / m * X.T @ error
        weights -= learning_rate * gradient

        if verbose and (iteration + 1) % 100 == 0:
            print(f"Error at epoch {iteration + 1}: {mean_squared_error}")

        if iteration > 0 and (error_history[-2] - mean_squared_error) <= convergence_threshold:
            break

    return weights, error_history, iteration + 1
