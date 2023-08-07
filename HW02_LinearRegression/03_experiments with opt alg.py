from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.animation as animation





np.random.seed(1337)

# Define the function to update the plot for every iteration
def update_plot(i):
    # Get the ith set of parameters
    params = params_list[i]
    # Calculate the predicted values using the parameters
    y_pred = params[0] * x + params[1]
    # Update the plot
    line.set_ydata(y_pred)
    ax.set_title(f"Epoch {i+1}")
    return line,
def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def mse_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """MSE Minimization Regression"""
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat = m * x + b
    # Get loss MSE
    mse = (np.square(y - y_hat)).mean()
    return mse



if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")


    # Initial guess of the parameters: [2, 2] (m, b).
    # It doesn’t have to be accurate but simply reasonable.
    initial_guess = np.array([5, -3])

    mse_list = []  # Create an empty list to store the MSE values
    params_list = []


    # Maximizing the probability for point to be from the distribution
    results = minimize(
        mse_regression,
        initial_guess,
        args=(x, y,),
        # method="Nelder-Mead",
        # method="COBYLA",
        # method="dogleg",
        method='BFGS',
        options={"disp": True, "return_all": True},
        #callback=lambda xk: mse_list.append(mse_regression(xk, x, y))
        callback=lambda xk: (params_list.append(xk), mse_list.append(mse_regression(xk, x, y)))

    )



    print(results)
    print("Parameters: ", results.x)
    print("MSE for each iteration:", mse_list)  # Print the list of MSE values after all iterations
    print(len(mse_list))
    print(params_list)




    # Plot results
    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = results.x[0] * xx + results.x[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(xx, yy, "r--.", label="MSE")
    ax.legend(loc="best")

    plt.savefig("mse_regression.png")

    highest_mse = np.amax(mse_list)
    lowest_mse = np.amin(mse_list)
    print(f'Highest MSE: {highest_mse}')
    print(f'Lowest MSE: {lowest_mse}')

    # Set the Seaborn style
    sns.set_style('whitegrid')





    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(mse_list)), mse_list, color='blue')
    ax.set_title('Loss function value over iterations', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MSE', fontsize=14)

    # Set the font size and style of the text
    ax.text(0.8, 0.9, f'Highest MSE: {highest_mse:.2f}\nLowest MSE: {lowest_mse:.2f}',
            transform=ax.transAxes, ha='center', va='center', fontsize=12, fontstyle='italic')

    # Add a grid to the plot
    ax.grid(True)
    plt.savefig("mse_loss.png")



