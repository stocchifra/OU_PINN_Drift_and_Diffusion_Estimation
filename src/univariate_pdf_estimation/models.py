# univariate_pdf/src/models.py

from flax import linen as nn
import jax.numpy as jnp

class ANN(nn.Module):
    hidden_layers: list

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the ANN.

        Parameters:
        - x: Input tensor.

        Returns:
        - Output tensor with non-negative values.
        """
        for feature in self.hidden_layers:
            x = nn.Dense(feature)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        x = nn.softplus(x)  # Ensuring the output is non-negative
        return x

def build_model(hidden_layers):
    """
    Build and return an ANN model based on the hidden layer configuration.

    Parameters:
    - hidden_layers: List of integers representing the size of each hidden layer.

    Returns:
    - model: Initialized ANN model instance.
    """
    model = ANN(hidden_layers=hidden_layers)
    return model
