"""
Dataset references:
http://pjreddie.com/media/files/mnist_train.csv
http://pjreddie.com/media/files/mnist_test.csv
"""


from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1337)


class MnistMlp(torch.nn.Module):

    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int) -> None:
        super().__init__()

        # number of nodes (neurons) in input, hidden, and output layer
        self.wih = torch.nn.Linear(in_features=inputnodes, out_features=hiddennodes)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hiddennodes)
        self.dropout1 = torch.nn.Dropout(p=0.5)  # added dropout layer
        self.hidden2 = torch.nn.Linear(in_features=hiddennodes, out_features=150)
        self.bn2 = torch.nn.BatchNorm1d(num_features=150)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.hidden3 = torch.nn.Linear(in_features=150, out_features=50)
        self.bn3 = torch.nn.BatchNorm1d(num_features=50)
        self.dropout3 = torch.nn.Dropout(p=0.2)
        self.who = torch.nn.Linear(in_features=50, out_features=outputnodes)
        self.bn4 = torch.nn.BatchNorm1d(num_features=outputnodes)
 #       self.activation = torch.nn.Sigmoid()
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.wih(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)  # added dropout layer
        out = self.hidden2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.hidden3(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.dropout3(out)
        out = self.who(out)
        out = self.bn4(out)
        return out

# класс обертка
class MnistDataset(Dataset):
    
    def __init__(self, filepath: Path) -> None:
        super().__init__()

        self.data_list = None
        with open(filepath, "r") as f:
            self.data_list = f.readlines()

        # conver string data to torch Tensor data type
        self.features = []
        self.targets = []
        for record in self.data_list:
            all_values = record.split(",")
            features = np.asfarray(all_values[1:])
            target = int(all_values[0])
            self.features.append(features)
            self.targets.append(target)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float) / 255.0
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.long)
        # print(self.features.shape)
        # print(self.targets.shape)
        # print(self.features.max(), self.features.min())

    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


if __name__ == "__main__":
    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NN architecture:
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate is 0.1
    learning_rate = 0.1
    # batch size
    #batch_size = 10

    batch_size = 100

    # number of epochs
    #epochs = 3
    epochs = 30

    # Load mnist training and testing data CSV file into a datasets
    train_dataset = MnistDataset(filepath="./mnist_train.csv")
    test_dataset = MnistDataset(filepath="./mnist_test.csv")

    # Make data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # Define NN
    model = MnistMlp(inputnodes=input_nodes, 
                     hiddennodes=hidden_nodes, 
                     outputnodes=output_nodes)
    # Number of parameters in the model
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device=device)
    
    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    train_loss_arr = []
    train_accuracy_arr = []
    test_loss_arr = [] # 1 раз на епоху брать
    test_accuracy_arrr = []

    for epoch in range(epochs):
        ##### Training! #####
        model.train()

        train_batch_loss = []
        correct = 0

        for batch_idx, (features, target) in enumerate(train_loader): #идем побачам
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target) #лосс каждого бача на епохе

            pred = output.argmax(dim=1, keepdims=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            train_batch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(features), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        train_loss_arr.append(np.mean(np.array(train_batch_loss))) #256 бачей за епоху дали лоссы и мы взяли их среднее арифметическое
        train_accuracy_arr = [100. * correct / len(train_loader.dataset)]

        ##### Testing! ##### перевожу оценку на каждую епоху
        model.eval()
        test_loss = 0
        correct = 0
        with torch.inference_mode():
            for features, target in test_loader:
                features, target = features.to(device), target.to(device)
                output = model(features)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)# средний лосс за епоху
        test_loss_arr.append(test_loss)
        test_accuracy_arr = [100. * correct / len(test_loader.dataset)]

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    #вывожу архитектуру
    print(model)

    ##### Save Model! #####
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "mnist_001.pth")

    # Plot and label the training and validation loss values
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(range(epochs), np.log(train_loss_arr), label='Training Loss')
    plt.plot(range(epochs), np.log(test_loss_arr), label='Test Loss')

    ax.legend(loc="best")

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("loss3.png")

