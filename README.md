# OU_PINN_Drift_and_Diffusion_Estimation

This repository uses Physics-Informed Neural Networks (PINNs) to estimate the probability density function (PDF), drift, and diffusion of univariate and bivariate Ornstein-Uhlenbeck (OU) processes. It combines deep learning with physics constraints to model stochastic dynamics, specifically using the Fokker-Planck equation as the physics constraint.

## Introduction

The Ornstein-Uhlenbeck process is a stochastic process frequently used to model mean-reverting behavior in fields such as finance, physics, and biology. The goal of this repository is to accurately estimate the PDF, drift, and diffusion parameters of both univariate and bivariate OU processes using neural networks.

### Fokker-Planck Equation

The Fokker-Planck equation describes the time evolution of the probability density function of the state variable of a stochastic process. For an OU process, the Fokker-Planck equation serves as a key constraint to enforce the physical realism of the model's outputs.

### Univariate Case

**Ornstein-Uhlenbeck Process:**

$$
dX_t = \theta (\mu - X_t) dt + \sigma dW_t
$$

where:
- $\theta$: Rate of mean reversion
- $\mu$: Long-term mean
- $\sigma$: Volatility
- $dW_t$: Wiener process (Brownian motion)

**Fokker-Planck Equation:**

$$
\frac{\partial P(x,t)}{\partial t} = -\frac{\partial}{\partial x} \left[\theta (\mu - x) P(x,t)\right] + \frac{1}{2} \frac{\partial^2}{\partial x^2} \left[\sigma^2 P(x,t)\right]
$$

### Bivariate Case

**Bivariate Ornstein-Uhlenbeck Process:**

$$
dX_t = \theta (\mu - X_t) dt + \sigma dW_t
$$

where the parameters now consider interactions between two dimensions.

**Bivariate Fokker-Planck Equation:**

$$
\frac{\partial P(x_1, x_2, t)}{\partial t} = -\nabla \cdot \left[\mathbf{A}(x) P(x_1, x_2, t)\right] + \frac{1}{2} \nabla \nabla : \left[\mathbf{B} P(x_1, x_2, t)\right]
$$

where:
- $\mathbf{A}(x)$ represents the drift vector.
- $\mathbf{B}$ represents the diffusion matrix.

## 1. Univariate Probability Density Function Estimation

This section provides a tutorial to run the training and testing of models for univariate PDF estimation.

### Training the Model

You can train either an MLP or PINN model for estimating the PDF of the univariate OU process. By default, the hyperparameters are optimized for the PINN model. If you want to use the optimized hyperparameters for the MLP model, use the specified MLP parameters.

### Command to Train the Model:

```bash
cd src/univariate_pdf_estimation
python training.py --model_type 'PINN' --epochs 87 --batch_size 913 --num_datasets 35 --num_paths 10 --T 4.0 --dt 0.01 --alpha 0.806 --beta 0.108 --hidden_layers '256,128,64' --patience 16 --min_delta 1e-4 --learning_rate 0.0007385 --weight_decay 0.00010097 --log_interval 100
````
The default hyperparameters are optimized for the PINN model, but users can choose each parameter as they prefer. If you want the optimized hyperparameters for the MLP, you can use the following:

```bash
cd src/univariate_pdf_estimation
python training.py --model_type 'MLP' --epochs 90 --batch_size 517 --num_datasets 35 --num_paths 10 --T 4.0 --dt 0.01 --hidden_layers '128,64,32' --patience 17 --min_delta 1e-4 --learning_rate 0.0009539 --weight_decay 1.0277e-05 --log_interval 100
````

## Testing the Univariate Probability Density Function Estimation

This section provides instructions on testing trained models on various parameter combinations of the Ornstein-Uhlenbeck (OU) process. The testing script evaluates the model's performance by calculating the Mean Squared Error (MSE) between the predicted and actual probability density functions generated from the OU process.

### Testing the Model

You can test either the `PINN` or `MLP` model using the following command. The default hyperparameters provided are optimized for the `PINN` model. If you trained your model using different hyperparameters, ensure to use the same configuration for testing.

### Command to Test the PINN Model:

```bash
cd src/univariate_pdf_estimation
python testing.py --model_type 'PINN' --num_combinations 150 --T 3.5 --dt 0.01 --num_paths 1000 --epochs 87 --batch_size 913 --alpha 0.806 --beta 0.108 --hidden_layers '256,128,64' --patience 16 --min_delta 1e-4 --learning_rate 0.0007385 --weight_decay 0.00010097 --log_interval 100
````
### Command to Test the MLP Model:
```bash
cd src/univariate_pdf_estimation
python testing.py --model_type 'MLP' --num_combinations 150 --T 3.5 --dt 0.01 --num_paths 1000 --epochs 90 --batch_size 517 --hidden_layers '128,64,32' --patience 17 --min_delta 1e-4 --learning_rate 0.0009539 --weight_decay 1.0277e-05 --log_interval 100
````

