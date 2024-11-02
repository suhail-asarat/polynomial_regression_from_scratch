import numpy as np
import matplotlib.pyplot as plt
import functions  # Ensure this module name matches your refactored functions file

# Set parameters for the data generation
num_samples = 500  # Number of data points
coefficients = [0, -1, 0, 0.3]  # Coefficients of the polynomial
x_min, x_max = -10, 10  # Range of x values
noise_level = 30  # Standard deviation of the noise

# Generate synthetic polynomial data
X, y = functions.generate_polynomial_data(num_samples, coefficients, x_min, x_max, noise_level)

# Plot the generated data alongside the polynomial without noise for comparison
x = X[:, 1]  # Extract the x values from the design matrix
y_hat_initial = X @ np.array(coefficients).reshape(-1, 1)  # Compute the initial predictions
functions.plot_data_predictions(x, y, y_hat_initial.flatten(), label_pred='Initial Prediction')
plt.title('Data Generation with Initial Prediction')

# Initialize the weights for polynomial regression model
initial_weights = functions.initialize_weights(X.shape[1], zero_init=False)

# Display the initial weights
print('Initial weights shape:', initial_weights.shape)
print('Initial weights:', initial_weights)

# Visualize the predictions using the initial weights
y_hat_zero = X @ initial_weights
functions.plot_data_predictions(x, y, y_hat_zero.flatten(), label_pred='Initial Model Prediction')
plt.title('Initial Model Prediction')

# Perform gradient descent to optimize the weights
final_weights, error_history, epochs = functions.perform_gradient_descent(X, y, initial_weights)

# Display the optimization results
print('Final weights:', final_weights)
print('Initial error:', error_history[0])
print('Final error:', error_history[-1])
print('Total epochs:', epochs)

# Plot the final predictions after training the model
y_hat_final = X @ final_weights
functions.plot_data_predictions(x, y, y_hat_final.flatten(), label_pred='Final Prediction')
plt.title('Model Prediction After Training')
