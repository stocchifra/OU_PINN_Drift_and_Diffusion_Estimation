# Utility functions for saving and loading model parameters

import numpy as np
import pickle
import jax.numpy as jnp


def initialize_model(rng_key, input_dim, hidden_layers):
    """
    Initialize the ANN model.

    Parameters:
    - rng_key: Random key for initialization.
    - input_dim: Dimension of the input.
    - hidden_layers: List of hidden layer sizes.

    Returns:
    - model: Initialized model.
    - params: Model parameters.
    """
    model = build_model(hidden_layers)
    dummy_input = jnp.ones((1, input_dim))
    params = model.init(rng_key, dummy_input)['params']
    return model, params

def to_state_dict(params):
    """
    Convert model parameters to a state dictionary.

    Parameters:
    - params: Model parameters.

    Returns:
    - State dictionary of model parameters.
    """
    return params

def from_state_dict(state_dict, params):
    """
    Load model parameters from a state dictionary.

    Parameters:
    - state_dict: State dictionary of model parameters.
    - params: Initialized model parameters.

    Returns:
    - Model parameters loaded from the state dictionary.
    """
    return state_dict

def save_params(params, path_item: str):
    """
    Save model parameters to a file.

    Parameters:
    - params: Model parameters to be saved.
    - path_item: File path where parameters should be saved.
    """
    with open(path_item, 'wb') as f:
        pickle.dump(to_state_dict(params), f)

def load_params(params_initialized, path_item: str):
    """
    Load model parameters from a file.

    Parameters:
    - params_initialized: Initialized model parameters.
    - path_item: File path from where parameters should be loaded.

    Returns:
    - Loaded model parameters.
    """
    with open(path_item, 'rb') as f:
        loaded_dict = pickle.load(f)
    return from_state_dict(loaded_dict, params_initialized)



def build_model(config):
    """
    Build the ANN model based on the configuration provided.

    Parameters:
    - config: Configuration object containing model parameters.

    Returns:
    - model: Initialized ANN model instance.
    """
    # Initialize the model with the hidden layers specified in the config
    model = ANN(hidden_layers=config.hidden_layers)
    return model

# Function to load the trained model parameters
def load_trained_model(weights_file, config):
    """
    Loads the trained model using the saved weights and hyperparameters.
    
    Parameters:
    - weights_file: Path to the file where model weights are stored.
    - config: Configuration object containing model parameters.
    
    Returns:
    - model: ANN model instance.
    - params: Loaded model parameters.
    """
    # Build the model architecture from config
    model = build_model(config)
    
    # Initialize model parameters
    rng_key = jax.random.PRNGKey(config.seed)  # Use config.seed to generate a reproducible key
    _, params_initialized = initialize_model(rng_key, config.ann_in_dim, config.hidden_layers)
    
    # Load the trained model parameters from file
    params_trained = load_params(params_initialized, weights_file)  # Loading the actual trained parameters
    
    return model, params_trained  # Return the model and the loaded parameters


def generate_random_parameters_univariate(seed=0):
    """
    Generates random combinations of mu, theta, and sigma for a univariate OU process.
    
    Parameters:
    - seed (int): Seed for random number generation for reproducibility.

    Returns:
    - mu (float): Random value for mu in the range [-1, 1].
    - theta (float): Random positive value for theta in the range [0.1, 2.0].
    - sigma (float): Random positive value for sigma representing volatility in the range [0.1, 1.0].
    """
    if seed is not None:
        np.random.seed(seed)
    mu = np.random.uniform(-1, 1)
    theta = np.random.uniform(0.1, 2.0)
    sigma = np.random.uniform(0.1, 1.0)
    return mu, theta, sigma


def decode_hidden_layers(encoded_layers):
    # Convert encoded string back to tuple of integers
    return tuple(map(int, encoded_layers.split("-")))


