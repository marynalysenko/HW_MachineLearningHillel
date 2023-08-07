#Experiment with non-linear data, for example: y = 2 * x**2 + x + 3.5 + noise

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

np.random.seed(1337)

sample_size = 100
def get_data(nsamples: int = sample_size) -> Tuple[np.array, np.array]:
    x = np.linspace(-2500, 1500, nsamples)
    y =  2 * x**2 + x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def mse_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """MSE Minimization Regression"""
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat =  m * x**2 + x + b
    # Get loss MSE
    mse = (np.square(y - y_hat)).mean()
    return mse


def rmse_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """RMSE Minimization Regression"""
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat =   m * x**2 + x + b
    # Get loss RMSE
    rmse = np.sqrt(((y - y_hat) ** 2).mean())
    return rmse


def mae_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """MAE Minimization Regression"""
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat =   m * x**2 + x + b
    # Get loss MAE
    mae = np.abs(y - y_hat).mean()
    return mae


def regression(x, y, y_true, loss_function, loss_name):
    loss_list = []

    # Initial guess of the parameters: [2, 2] (m, b).
    # It doesn’t have to be accurate but simply reasonable.

    initial_guess = np.array([5, -3])

    results = minimize(
        loss_function,
        initial_guess,
        args=(x, y),
        method="Nelder-Mead",
        options={"disp": True, "return_all": True},
        callback=lambda xk: loss_list.append(loss_function(xk, x, y)),
    )
    print(results)
    print(f"Parameters ({loss_name}): ", results.x)
    print(f"{loss_name} for each iteration ({loss_name}):", loss_list)

    # Plot results
    xx = np.linspace(np.min(x), np.max(x), sample_size)
    yy = results.x[0] * x ** 2 + x +  results.x[1]


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot regression line
    ax[0].plot(x, y, "o", label="data")
    ax[0].plot(x, y_true, "b-", label="True")
    ax[0].plot(xx, yy, "r--.", label=loss_name)
    ax[0].legend(loc="best")
    ax[0].set_title(f"{loss_name} Regression")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    # Plot loss function value over iterations
    highest_loss = np.amax(loss_list)
    lowest_loss = np.amin(loss_list)

    print(f'Highest loss: {highest_loss}')
    print(f'Lowest loss: {lowest_loss}')

    ax[1].plot(range(len(loss_list)), loss_list, color="blue")
    ax[1].set_title("Loss function value over iterations")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(loss_name)
    ax[1].text(
        0.8,
        0.9,
        f"Highest {loss_name}: {highest_loss:.2f}\nLowest {loss_name}: {lowest_loss:.2f}",
        transform=ax[1].transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontstyle="italic",
    )
    ax[1].grid(True)

    # Save plots
    plt.savefig(f"{loss_name}_plots.png")



if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")



    # Plot regression for different loss functions
    regression(x, y, y_true, mse_regression, "MSE")
    regression(x, y, y_true, mae_regression, "MAE")
    regression(x, y, y_true, rmse_regression, "RMSE")
