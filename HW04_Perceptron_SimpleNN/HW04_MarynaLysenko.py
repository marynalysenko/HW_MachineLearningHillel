from typing import Tuple
import torch
from torch import nn  # nn contains all of PyTorch's building blocks for neural networks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(1337)
torch.manual_seed(314)

m  = 2
b  = 3.5

def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = m * x + b
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def plot_predictions(model, X, y, title):
    with torch.no_grad():
        y_pred = model(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data, predicted values, and true line
    ax.plot(X, y, "ok", label="data")
    ax.plot(X, y_pred, color='c', label="predicted")
    ax.plot(X, m * X + b, color="b", label="True")

    ax.set_title(title)
    ax.legend(loc="best")
    plt.savefig(title+".png")


# Create a Linear Regression model class
class LinearRegressionModel(
    nn.Module):  # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1,  # <- start with random weights (this will get adjusted as the model learns)
                        dtype=torch.float),  # <- PyTorch loves float32 by default
            requires_grad=True)  # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(
            torch.randn(1,  # <- start with random bias (this will get adjusted as the model learns)
                        dtype=torch.float),  # <- PyTorch loves float32 by default
            requires_grad=True)  # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias  # <- this is the linear regression formula (y = m*x + b)


if __name__ == "__main__":
    # Get available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device = }")

    # Check PyTorch version
    print(f"Using {torch.__version__ = }")

    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    x, y_true, y = torch.tensor(x), torch.tensor(y_true), torch.tensor(y)

    # Plot and investigate data
    # Set seaborn style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "ok", label="data")
    ax.set_title("True Data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # Create train/test split
    # 80% of data used for training set, 20% for testing

    train_split = int(0.8 * len(x))
    X_train, y_train = x[:train_split], y[:train_split]
    X_test, y_test = x[train_split:], y[train_split:]

    print(len(x), len(y))
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))

    # Create an instance of the model (this is a subclass of
    # nn.Module that contains nn.Parameter(s))
    model_0 = LinearRegressionModel()

    # Check the nn.Parameter(s) within the nn.Module
    # subclass we created
    print(f"{list(model_0.parameters()) = }")
    # List named parameters
    print(f"{model_0.state_dict() = }")

    # Print the weights and biases before training
    print("Initial weights:", model_0.weights.data.item())
    print("Initial biases:", model_0.bias.data.item())

    # Create the loss function
    loss_fn = nn.L1Loss()  # L1Loss loss is same as MAE

    # Create the optimizer
    # ``parameters`` of target model to optimize
    # ``learning rate`` (how much the optimizer should change parameters
    # at each step, higher=more (less stable), lower=less (might take a long time))
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # Set the number of epochs (how many times
    # the model will pass over the training data)
    epochs = 100

    # Initialise array to store the training and validation losses
    train_loss_arr = []
    test_loss_arr = []
    for epoch in range(epochs):
        ### Training

        # Put model in training mode (this is the default state of a model)
        model_0.train()

        # 1. Forward pass on train data using the forward() method inside
        y_pred = model_0(X_train)

        # 2. Calculate the loss (how different are our models predictions
        # to the ground truth)
        loss = loss_fn(y_pred, y_train)
        train_loss_arr.append(loss.item())
        #print(train_loss)

        # 3. Zero grad of the optimizer
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Progress the optimizer
        optimizer.step()

        ### Testing

        # Put the model in evaluation mode
        model_0.eval()

        with torch.inference_mode():
            # 1. Forward pass on test data
            test_pred = model_0(X_test)

            # 2. Caculate loss on test data
            test_loss = loss_fn(test_pred, y_test.type(
                torch.float))  # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
            test_loss_arr.append(test_loss.item())

            # Print out what's happening
            if epoch % 10 == 0:
                # List named parameters  print(f"{model_0.state_dict() = }")

                state_dict = model_0.state_dict()
                weights = state_dict['weights']
                bias = state_dict['bias']
                #
                #  Extract the weights and biases from the model and print them
                #
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} | Weights: {weights.item()} | Bias: {bias.item()}")


    print(f"Last Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} | Weights: {weights.item()} | Bias: {bias.item()}")
    print(len(range(epochs)), len(train_loss_arr), len(test_loss_arr))


    # Plot and label the training and validation loss values
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(range(epochs), np.log(train_loss_arr), label='Training Loss')
    plt.plot(range(epochs), np.log(test_loss_arr), label='Test Loss')

    ax.legend(loc="best")

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("loss.png")


    # Put the model in evaluation mode
    model_0.eval()

    plot_predictions(model_0, X_train, y_train, "Predicted vs. True Values (Training Data)")
    plot_predictions(model_0, X_test, y_test, "Predicted vs. True Values (Test Data)")



