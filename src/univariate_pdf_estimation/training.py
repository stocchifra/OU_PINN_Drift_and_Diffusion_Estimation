# univariate_pdf/src/training.py

import argparse
import jax
from flax.training import train_state
import optax
import time
from models import build_model
from losses import loss_MLP, loss_PINN
from jax import grad, vmap, jit as jjit
import jax.numpy as jnp
from losses import errors_fp
from data_generation import generate_training_data_with_random_params
from utils import save_params, decode_hidden_layers
from config import Config
from visualization import plot_training_history




# Dictionary to map loss functions
loss_fn = {
    "MLP": lambda fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu, efunc: loss_MLP(fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu, efunc),
    "PINN": lambda fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu, efunc, alpha, beta: loss_pinn(fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu, efunc, alpha, beta)
}

# Error function for the Fokker-Planck equation
error_fn = {"FP_OU": errors_fp}


def ann_gen(config):
    """
    Generate an ANN model and optimizer based on the configuration.

    Parameters:
    - config: Configuration object.

    Returns:
    - model: ANN model instance.
    - tx: Optimizer instance.
    """
    model = build_model(config.hidden_layers)
    lr_schedule = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.decay_steps,
        decay_rate=config.decay_rate,
    )
    tx = optax.chain(
        optax.add_decayed_weights(config.weight_decay),
        optax.adam(
            learning_rate=lr_schedule, b1=config.beta1, b2=config.beta2, eps=config.eps
        )
    )
    return model, tx

# Calibration function to set up the training step
def calibration(config):
    """
    Set up the training step for model calibration.
    """
    ann, tx = ann_gen(config)

    ofunc = loss_fn[config.loss_str]
    efunc = error_fn[config.data_source]

    @jjit
    def train_step(state, batch_x, batch_y, x_mesh, x_val, y_val, batch_theta_values, batch_sigma_values, batch_mu_values):
        
        def loss_fn(params):
            def fn(x):
                result = ann.apply({'params': params}, x)
                return result

            if config.loss_str == "PINN":
                loss, metrics = ofunc(fn, batch_x, batch_y, x_mesh, x_val, y_val, batch_theta_values, batch_sigma_values, batch_mu_values, efunc, config.alpha, config.beta)
                return loss, metrics
            else:
                loss, metrics = ofunc(fn, batch_x, batch_y, x_mesh, x_val, y_val, batch_theta_values, batch_sigma_values, batch_mu_values, efunc)
                return loss, metrics

        params = state.params
        (loss, metric), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        state = state.apply_gradients(grads=grads)
        return state, loss, metric

    return train_step

# Batch Generator Function
def batch_generator(x_data, y_data, x_mesh, mu_values, theta_values, sigma_values, batch_size):
    """
    Generator function to yield batches of data including mu, theta, and sigma.

    Parameters:
    - x_data: The input features dataset.
    - y_data: The target values dataset.
    - mu_values: Long-term mean values.
    - theta_values: Drift coefficient values.
    - sigma_values: Diffusion coefficient values.
    - batch_size: The size of each batch.

    Yields:
    - A batch of data including x, y, mu, theta, and sigma.
    """
    for i in range(0, len(x_data), batch_size):
        yield (x_data[i:i + batch_size],
               y_data[i:i + batch_size],
               x_mesh[i:i + batch_size],
               mu_values[i:i + batch_size],
               theta_values[i:i + batch_size],
               sigma_values[i:i + batch_size])


def train_model(config, data, theta_values, sigma_values, mu_values, batch_size=32):
    """
    Train the model using the provided data and configuration.

    Parameters:
    - config: Configuration object.
    - data: Training data tuple (x_train, y_train, x_val, y_val).
    - theta_values, sigma_values, mu_values: OU process parameters.
    - batch_size: Batch size for training.

    Returns:
    - params: Trained model parameters.
    - metrics: Dictionary of training metrics.
    """
    x_train, y_train, x_mesh, x_val, y_val = data
    ann, tx = ann_gen(config)
    train_step = calibration(config)
    key = jax.random.PRNGKey(config.seed)
    key, key_init = jax.random.split(key, 2)
    dummy = jnp.ones((1, config.ann_in_dim))

    state = train_state.TrainState.create(
        apply_fn=ann.apply,
        params=ann.init(key_init, dummy)['params'],
        tx=tx
    )

    metrics = {'loss': [], 'e_acc': [], 'e_pde': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_params = None
    patience_counter = 0

    print("------ Training {} start ------".format(config.loss_str))
    start_time = time.time()

    for epoch in range(config.num_epochs):
        batch_gen = batch_generator(x_train, y_train, x_mesh, mu_values, theta_values, sigma_values, batch_size)
        epoch_loss = 0
        epoch_metrics = {k: 0 for k in ['e_acc', 'e_pde', 'val_acc']}
        num_batches = 0
        
        for batch_x, batch_y, batch_x_mesh, batch_mu_values, batch_theta_values, batch_sigma_values in batch_gen:
            state, loss, metric = train_step(state, batch_x, batch_y, batch_x_mesh, x_val, y_val, batch_theta_values, batch_sigma_values, batch_mu_values)
            epoch_loss += loss
            for k, v in metric.items():
                epoch_metrics[k] += v
            num_batches += 1

        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        metrics['loss'].append(epoch_loss)
        metrics['e_acc'].append(epoch_metrics['e_acc'])
        metrics['e_pde'].append(epoch_metrics['e_pde'])
        metrics['val_acc'].append(epoch_metrics['val_acc'])

        val_loss = epoch_metrics['val_acc']

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_params = state.params
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss}")
            state = state.replace(params=best_params)
            break

    print("----- Training {} completed in {:0.2f} sec ------".format(config.loss_str, time.time() - start_time))
    return state.params, metrics



def main(args):
    # Example configuration setup
    decoded_hidden_layers = args.hidden_layers
    
    if args.model_type == "MLP":
        config = Config(
            seed=42,
            ann_in_dim=2,
            hidden_layers= decoded_hidden_layers,  # Example hidden layer sizes
            loss_str=args.model_type,  # Choose between 'MLP' or 'PINN'
            data_source='FP_OU',
            num_epochs=args.epochs,
            log_intervala=args.log_interval,
            patience=args.patience,
            min_delta=args.min_delta,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size
        )
    else:
        config = Config(
            seed=42,
            ann_in_dim=2,
            hidden_layers=decoded_hidden_layers,  # Example hidden layer sizes
            loss_str=args.model_type,  # Choose between 'MLP' or 'PINN'
            data_source='FP_OU',
            num_epochs=args.epochs,
            log_interval=args.log_interval,
            alpha=args.alpha,  # Only used for PINN
            beta=args.beta,   # Only used for PINN
            patience=args.patience,
            min_delta=args.min_delta,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size
        )

    # Generate training data with random parameters
    inputs, targets, mus, sigmas, thetas = generate_training_data_with_random_params(args.num_datasets, args.num_paths, args.T, args.dt)

    # Split data into training and validation sets
    train_size = int(0.8 * len(inputs))

    x_train = inputs[:train_size]
    y_train = targets[:train_size]
    mu_train = mus[:train_size]
    sigma_train = sigmas[:train_size]
    theta_train = thetas[:train_size]

    x_val_train = inputs[train_size:]
    y_val_train = targets[train_size:]

    data_train = (x_train, y_train, x_train, x_val_train, y_val_train)

    # Call the training function
    params, metrics = train_model(config, data_train, theta_train, sigma_train, mu_train, config.batch_size)

    # Optionally, save the trained parameters and/or metrics
    save_params(params, f"model_params_{config.loss_str}_estimation.pkl")
    
    # Print or visualize training results
    print("Training completed. Metrics:")
    print(metrics)
    
    plot_training_history(metrics, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network for univariate PDF estimation.")
    
    # Adding arguments for model type and configuration
    parser.add_argument('--model_type', type=str, choices=['MLP', 'PINN'], default='MLP',
                        help="Type of model to train: 'MLP' for Multi-Layer Perceptron or 'PINN' for Physics-Informed Neural Network.")
    parser.add_argument('--epochs', type=int, default=87, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=913, help="Batch size for training.")
    parser.add_argument('--num_datasets', type=int, default=35, help="Number of datasets to generate for training.")
    parser.add_argument('--num_paths', type=int, default=10, help="Number of paths per dataset.")
    parser.add_argument('--T', type=float, default=4.0, help="Total time for the OU process simulation.")
    parser.add_argument('--dt', type=float, default=0.01, help="Time step for the OU process simulation.")
    parser.add_argument('--alpha', type=float, default=0.8060688949778174, help="Weight for the accuracy error in PINN loss.")
    parser.add_argument('--beta', type=float, default=0.10847539042978625, help="Weight for the PDE residual error in PINN loss.")
    parser.add_argument('--hidden_layers', type=str, default='256,128,64', 
                        help="Comma-separated sizes of hidden layers (e.g., '128,64,32').")
    parser.add_argument('--patience', type=int, default=16, help="Number of epochs to wait for improvement before stopping early.")
    parser.add_argument('--min_delta', type=float, default=1e-4, help="Minimum change in validation loss to qualify as an improvement.")
    parser.add_argument('--learning_rate', type=float, default=0.0007384984732477898, help="Learning rate for the optimizer.")
    parser.add_argument('--weight_decay', type=float, default=0.00010096684101720352, help="L2 regularization term (weight decay).")
    parser.add_argument('--log_interval', type=int, default=100, help="Interval for logging training progress.")


    # Parsing arguments from the command line
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
    

