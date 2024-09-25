# univariate_pdf/src/plotting.py

import matplotlib.pyplot as plt


def plot_all_paths_and_P_xt(t, X, P_xt, x_grid, dataset_index, mu, sigma, theta):
    """
    Plot the paths of the Ornstein-Uhlenbeck process and the corresponding probability density P(x,t) 
    at the last time step.
    
    Parameters:
    - t: Array of time points.
    - X: Array of generated paths.
    - P_xt: Array of P(x,t) values.
    - x_grid: Grid of x values.
    - dataset_index: Index of the current dataset (for labeling).
    - mu: Long-term mean (μ).
    - sigma: Volatility (σ).
    - theta: Rate of mean reversion (θ).
    """
    plt.figure(figsize=(14, 6))

    # Plot the first 3 paths
    plt.subplot(1, 2, 1)
    for i in range(X.shape[0]):
        plt.plot(t, X[i], alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Dataset {dataset_index + 1}: First 3 Paths (μ={mu:.2f}, σ={sigma:.2f}, θ={theta:.2f})')
    plt.grid(True)

    # Plot P(x,t) for the last time step
    plt.subplot(1, 2, 2)
    P_xt_last_time = P_xt[:, -1]  # Select the last time step
    plt.plot(x_grid, P_xt_last_time, label=f'P(x,t) at t={t[-1]:.2f}')
    plt.xlabel('x')
    plt.ylabel('P(x,t)')
    plt.title(f'Dataset {dataset_index + 1}: P(x,t) (μ={mu:.2f}, σ={sigma:.2f}, θ={theta:.2f})')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_predictions_at_time_steps(x_grid, predictions, targets, t, time_steps):
    """
    Plot the predicted and actual probability densities P(x,t) at selected time steps.

    Parameters:
    - x_grid: Grid of x values.
    - predictions: Predicted P(x,t) values, reshaped as (len(x_grid), len(t)).
    - targets: Actual P(x,t) values, reshaped as (len(x_grid), len(t)).
    - t: Array of time points.
    - time_steps: List of indices of time steps to plot.
    """
    time_labels = [f't = {round(t[i], 2)}' for i in time_steps]

    plt.figure(figsize=(12, 8))

    # Plot P(x,t) for selected time steps
    for i, step in enumerate(time_steps):
        plt.plot(x_grid, predictions[:, step], label=f'Predicted {time_labels[i]}', linestyle='--')
        plt.plot(x_grid, targets[:, step], label=f'Actual {time_labels[i]}', linestyle='-')

    plt.xlabel('x')
    plt.ylabel('P(x,t)')
    plt.title('Probability Density Function P(x,t) at Different Time Steps')
    plt.legend()
    plt.grid(True)
    plt.show()
