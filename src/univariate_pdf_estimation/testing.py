import argparse
import time
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import generate_random_parameters_univariate, decoded_hidden_layers
from models import load_trained_model
from data_generation import generate_ou_process_data
from config import Config   

def test_model(config, model_type, num_combinations=150, T=3.5, dt=0.01, num_paths=1000):
    """
    Test the trained model on multiple combinations of OU process parameters.

    Parameters:
    - config: Configuration object with settings for the model.
    - model_type: Type of model to load ('PINN' or 'MLP').
    - num_combinations: Number of parameter combinations to generate and test on.
    - T: Total time for the OU process simulation.
    - dt: Time step for the OU process simulation.
    - num_paths: Number of paths to simulate for each combination.

    Returns:
    - Dictionary containing the MSE scores for each combination.
    """
    # Generate specified number of combinations of mu, theta, and sigma
    params_dict = {}
    for i in range(num_combinations):
        mu, theta, sigma = generate_random_parameters_univariate()
        params_dict[f'combination_{i+1}'] = {'mu': mu, 'theta': theta, 'sigma': sigma}

    # Load the trained model parameters based on model type
    weights_file = f"model_params_{model_type}_p_estimation.pkl"
    model, params_trained = load_trained_model(weights_file, config)

    # Initialize a list to store MSE scores
    mse_scores = []

    start_time = time.time()

    # Test the model on each parameter combination
    for key, param_set in params_dict.items():
        mu = param_set['mu']
        theta = param_set['theta']
        sigma = param_set['sigma']
        
        # Generate the data using the OU process function
        X0 = 0.5  # Initial value
        inputs, targets, t, X, P_xt, x_grid = generate_ou_process_data(theta, mu, sigma, T, dt, num_paths)
        
        # Ensure inputs are correctly shaped
        inputs = jnp.asarray(inputs)
        
        # Predict using the trained model
        predictions = model.apply({'params': params_trained}, inputs)
        
        # Calculate MSE
        mse = mean_squared_error(targets, predictions)

        # Append the MSE score to the list
        mse_scores.append(mse)

        print(f"{key}: MSE = {mse}")

    # Calculate mean and standard deviation of MSE scores
    mse_array = np.array(mse_scores)
    mean_mse = np.mean(mse_array)
    std_mse = np.std(mse_array)

    print(f"Mean MSE {model_type} ({num_combinations} samples): {mean_mse}, Std. Dev. MSE: {std_mse}")
    print(f"Execution time for {model_type} ({num_combinations} samples): {time.time() - start_time:.2f} seconds")

    # Store results in a dictionary
    mse_results = {key: mse for key, mse in zip(params_dict.keys(), mse_scores)}

    return mse_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained neural network on various OU process parameters.")
    
    # Add arguments for the number of parameter combinations, T, dt, and num_paths
    parser.add_argument('--num_combinations', type=int, default=150,
                        help="Number of parameter combinations to generate and test on.")
    parser.add_argument('--T', type=float, default=3.5, help="Total time for the OU process simulation.")
    parser.add_argument('--dt', type=float, default=0.01, help="Time step for the OU process simulation.")
    parser.add_argument('--num_paths', type=int, default=1000, help="Number of paths to simulate for each combination.")
    parser.add_argument('--model_type', type=str, choices=['MLP', 'PINN'], default='PINN',
                        help="Type of model to load for testing: 'MLP' or 'PINN'.")
    

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


    
    args = parser.parse_args()

    decoded_hidden_layers = decoded_hidden_layers(args.hidden_layers)
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

    # Call the test_model function with user-defined and default parameters
    mse_results = test_model(config, args.model_type, args.num_combinations, args.T, args.dt, args.num_paths)
