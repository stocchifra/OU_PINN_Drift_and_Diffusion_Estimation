# univariate_pdf/src/data_generation.py

import numpy as np
from scipy.stats import gaussian_kde

def generate_ou_process_data(theta, mu, sigma, X0, T, dt, num_paths):
    """
    Generate time series data from an Ornstein-Uhlenbeck process and calculate the corresponding P(x,t).
    """
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps)
    X = np.zeros((num_paths, num_steps))
    X[:, 0] = X0

    for i in range(1, num_steps):
        dW = np.sqrt(dt) * np.random.randn(num_paths)
        X[:, i] = X[:, i-1] + theta * (mu - X[:, i-1]) * dt + sigma * dW

    x_grid = np.linspace(np.min(X), np.max(X), 100)
    P_xt = np.zeros((len(x_grid), num_steps))

    for i in range(1, num_steps):
        mean = X0 * np.exp(-theta * t[i]) + mu * (1 - np.exp(-theta * t[i]))
        variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t[i]))
        if variance > 1e-10:
            P_xt[:, i] = (1.0 / np.sqrt(2 * np.pi * variance)) * \
                         np.exp(-0.5 * ((x_grid - mean)**2) / variance)
        else:
            P_xt[:, i] = 0  # Avoid division by zero
    return t, X, P_xt, x_grid


def generate_training_data_with_random_params(num_datasets, num_paths, T, dt, seed=None):
    """Generates multiple datasets of OU process paths with random parameters."""
    if seed is None:
        seed = 42
    np.random.seed(seed)

    all_inputs = []
    all_targets = []
    all_mus = []
    all_sigmas = []
    all_thetas = []

    for dataset_index in range(num_datasets):
        mu = np.random.uniform(0.5, 1.5)
        sigma = np.random.uniform(0.2, 0.5)
        theta = np.random.uniform(0.5, 1.0)
        
        inputs, targets, t, X, P_xt, x_grid = generate_ou_process_data(theta, mu, sigma, 0, T, dt, num_paths)
        
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_mus.append(np.full(inputs.shape[0], mu))
        all_sigmas.append(np.full(inputs.shape[0], sigma))
        all_thetas.append(np.full(inputs.shape[0], theta))

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_mus = np.concatenate(all_mus, axis=0)
    all_sigmas = np.concatenate(all_sigmas, axis=0)
    all_thetas = np.concatenate(all_thetas, axis=0)
    
    return all_inputs, all_targets, all_mus, all_sigmas, all_thetas
