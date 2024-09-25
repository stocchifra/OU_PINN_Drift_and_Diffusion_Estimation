# univariate_pdf/src/losses.py

import jax.numpy as jnp
from jax import grad, vmap

def spatial_derivatives(fn, x):
    """
    Compute first and second order spatial derivatives of the function.

    Parameters:
    - fn: The function whose derivatives are to be computed.
    - x: Input data points.

    Returns:
    - dx1: First order derivatives with respect to x.
    - d2x1: Second order derivatives with respect to x.
    """
    f = lambda x: fn(x).sum()
    f_dK = lambda x: grad(f)(x).sum()
    dx = vmap(grad(f), 0)(x)
    #print(f"Computed first-order derivatives, dx shape: {dx.shape}")
    d2x = vmap(grad(f_dK), 0)(x)
    #print(f"Computed second-order derivatives, d2x shape: {d2x.shape}")
    dx1 = dx[:, 0]
    d2x1 = d2x[:, 0]
    #print(f"spatial_derivatives: dx1 shape: {dx1.shape}, d2x1 shape: {d2x1.shape}")
    return dx1, d2x1

def time_derivative(fn, t, x):
    """
    Compute the first order time derivative of the function.

    Parameters:
    - fn: The function whose derivatives are to be computed.
    - t: Time points.
    - x: Input data points.

    Returns:
    - dt: First order derivatives with respect to time.
    """
    t_expanded = jnp.expand_dims(t, axis=-1) if t.ndim == 1 else t
    x_expanded = x if x.ndim == 2 else jnp.expand_dims(x, axis=-1)
    combined = jnp.concatenate([t_expanded, x_expanded], axis=-1)
    f = lambda t: fn(combined).sum()
    dt = vmap(grad(f), 0)(t)
    #print(f"time_derivative: dt shape: {dt.shape}")
    return dt[:, 0]



# Loss functions
l_acc = lambda x, y, degree=2: (x.flatten() - y.ravel())**degree

def l_pde_fp(dt, dx, d2x, p, x, mu_values, sigma_values, theta_values, degree=2):
    """
    Compute the PDE residual loss for the Fokker-Planck equation.

    Parameters:
    - dt: First order derivatives with respect to time.
    - dx: First order derivatives with respect to space (x).
    - d2x: Second order derivatives with respect to space (x).
    - p: Probability density function values.
    - theta: Drift coefficient.
    - sigma: Diffusion coefficient.
    - x: Input data points.
    - mu: Long-term mean.
    - degree: Degree of the loss (default is 2).

    Returns:
    - Residual loss for the Fokker-Planck equation.
    """
    drift_term = theta_values * (mu_values - x[:, 0]) * dx - theta_values * p
    diffusion_term = 0.5 * sigma_values**2 * d2x
    residual = dt + drift_term - diffusion_term
    return jnp.abs(residual**degree)



# Error functions
def errors_fp(fn, x_train, y_train, x_mesh, x_val, y_val, theta_values, sigma_values, mu_values):
    """
    Compute the error metrics for the Fokker-Planck equation.

    Parameters:
    - fn: The function representing the model.
    - data: Training and validation data.
    - theta: Drift coefficient.
    - sigma: Diffusion coefficient.
    - mu: Long-term mean.

    Returns:
    - Dictionary of error metrics.
    """

    # Extract time and space dimensions
    x = x_mesh[:, 0:1]
    t = x_mesh[:, 1:]

    print(f"Calling spatial_derivatives with x shape: {x.shape}")
    dx, d2x = spatial_derivatives(fn, x_mesh)
    print(f"Calling time_derivative with t shape: {t.shape} and x shape: {x.shape}")
    dt = time_derivative(fn, t, x)

    #print(f"errors_fp: dx shape: {dx.shape}, d2x shape: {d2x.shape}, dt shape: {dt.shape}")

    e_acc = l_acc(fn(x_train).reshape(-1), y_train)
    #print(f"Calling l_pde_fp")
    e_pde = l_pde_fp(dt, dx, d2x, fn(x_mesh), x_mesh, mu_values, sigma_values, theta_values)

    #print(f"errors_fp: e_acc shape: {e_acc.shape}, e_pde shape: {e_pde.shape}")

    val_acc = l_acc(fn(x_val).reshape(-1), y_val)
    #print(f"errors_fp: val_acc shape: {val_acc.shape}")

    return {
        'e_acc': jnp.mean(e_acc).sum(),
        'e_pde': jnp.mean(e_pde).sum(),
        'val_acc': jnp.mean(val_acc).sum()
    }





# Loss functions
def loss_MLP(fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu, efunc):
    """
    Compute the MLP loss.

    Parameters:
    - fn: The function representing the model.
    - data: Training and validation data.
    - theta: Drift coefficient.
    - sigma: Diffusion coefficient.
    - mu: Long-term mean.
    - efunc: Error function to compute error metrics.

    Returns:
    - loss: Computed loss.
    - err: Dictionary of error metrics.
    """
    err = efunc(fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu)
    loss = err['e_acc']
    return loss, err


def loss_pinn(fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu, efunc, alpha, beta):
    """
    Compute the PINN loss.

    Parameters:
    - fn: The function representing the model.
    - data: Training and validation data.
    - theta: Drift coefficient.
    - sigma: Diffusion coefficient.
    - mu: Long-term mean.
    - efunc: Error function to compute error metrics.
    - alpha: Weight for accuracy error.
    - beta: Weight for PDE residual error.

    Returns:
    - loss: Computed loss.
    - err: Dictionary of error metrics.
    """
    err = efunc(fn, x_train, y_train, x_mesh, x_val, y_val, theta, sigma, mu)
    loss = alpha * err['e_acc'] + beta * err['e_pde']

    return loss, err


