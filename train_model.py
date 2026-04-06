"""
Claire Liu, Yu-Jing Wei
-----
Building a Convolutional Neural Network with PyTorch.
In this file, we define a convolutional neural network called
MyNetwork, which is designed to classify images of handwritten
digits from the MNIST dataset.
"""

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Constants / Hyperparameters
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5

# Lists to store training and test losses, and the number of training examples seen
train_losses = []
train_counter = []
test_losses = []
test_counter = []


class MyNetwork(nn.Module):
    """MyNetwork is a convolutional neural network that consists of
    two convolutional layers, a dropout layer, and two fully connected
    layers. It is designed to classify images of handwritten digits
    from the MNIST dataset.
    """

    def __init__(self):
        """Setup the layers of the network in the constructor"""

        # Call the constructor of the parent class (nn.Module)
        super().__init__()

        # A convolution layer with 1 channel(greyscale) and 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # A dropout layer with a 0.5 dropout rate (50%)
        self.conv2_drop = nn.Dropout2d(p=0.5)

        # Fully connected Linear layer with 50 nodes
        # 28x28 -conv1-> 24x24 -pooling-> 12x12 -conv2-> 8x8 -pooling-> 4x4
        # 20 channels * 4 * 4 = 320 input features for the FC1 layer
        self.fc1 = nn.Linear(320, 50)

        # Fully connected Linear layer with 50 nodes
        # To identify 10 digits, we need 10 output nodes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """This function computes a forward pass for the network"""

        # conv1 → pooling → ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2 → dropout → pooling → ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Flattern output, 20 channels * 4 * 4 = 320
        # The -1 means that the size of that dimension is inferred from the other dimensions
        x = x.view(-1, 320)
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # Fully connected Linear layer with 10 nodes
        x = self.fc2(x)

        # log_softmax function applied to the output
        return F.log_softmax(x, dim=1)


def train_network(
        model, optimizer, train_loader, epoch):
    """This function trains the network for one epoch, and records the
    training loss and the index of the current batch for plotting"""
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # Clear the gradients
        optimizer.zero_grad()
        # perform a forward pass
        output = model(data)
        # compute the loss (Negative Log Likelihood Loss)
        loss = F.nll_loss(output, target)
        # perform a backward pass
        loss.backward()
        # update the weights
        optimizer.step()

        # Calculate accuracy
        train_losses.append(loss.item())
        # Record the index of the current batch for plotting
        train_counter.append(
            (batch_idx * BATCH_SIZE_TRAIN) + ((epoch - 1) * 60000)
        )


def test_network(model, test_loader):
    """This function tests the network on the test set, and records the
    average test loss and the accuracy for plotting"""
    # Set the model to evaluation mode, which turns off dropout
    model.eval()
    # keep track of number of correct predictions and total test loss
    correct = 0
    test_loss = 0

    # For testing, we don't calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target).item()

            # Get the index of the max log-probability,
            #  which is the predicted class
            pred = output.argmax(dim=1)

            # Compare the predicted class with the true class and count
            # the number of correct predictions
            correct += pred.eq(target).sum().item()

    # Calculate accuracy:
    # Average the test loss over all batches
    test_loss /= len(test_loader)
    # Calculate the percentage of correct predictions
    accuracy = 100. * correct / len(test_loader.dataset)
    # Append the average test loss to the list of test losses for plotting
    test_losses.append(test_loss)
    print(f'Test Accuracy = {accuracy:.2f}%')


def draw_n_test_images(test_loader, n):
    """This function draws n images from the test set and displays their
    ground truth labels"""
    # Get the first batch of test data and targets
    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)

    # Plot the first n images in the test set and their corresponding labels
    test_fig = plt.figure()
    test_fig.suptitle("MNIST Dataset Samples", fontsize=16)
    for i in range(n):
        plt.subplot(n//3 + (1 if n % 3 != 0 else 0), 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


def draw_training_and_test_loss():
    """This function plots the training and test losses over time"""
    loss_fig = plt.figure()
    loss_fig.suptitle("Training and Test Loss", fontsize=16)
    plt.plot(train_counter, train_losses, color='blue', label='Train loss')
    plt.scatter(test_counter, test_losses, color='red', zorder=5,
                label='Test loss')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend()
    plt.show()


def main(argv):
    """This function sets up the data, creates an instance of the network,
    and trains and tests the network for a number of epochs. It also plots
    the training and test losses"""

    # Training sets and load it into DataSet
    train_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(
                        './files/', train=True, download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                        ])),
                    batch_size=BATCH_SIZE_TRAIN, shuffle=True
                    )

    # Testing sets and load it into DataSet
    test_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(
                        './files/', train=False, download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                        ])),
                    batch_size=BATCH_SIZE_TEST, shuffle=False
                    )

    # Draw some test images with their ground truth labels
    draw_n_test_images(test_loader, n=6)

    # Create an instance of the network
    model = MyNetwork()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM)

    # Train the network and test it after each epoch
    for epoch in range(N_EPOCHS + 1):
        if epoch > 0:
            train_network(model, optimizer, train_loader, epoch)
        print(f"Epoch {epoch} finished.")
        test_network(model, test_loader)
        test_counter.append(epoch * 60000)

    # Save the model
    # state_dict() returns a dictionary containing model's params and values
    os.makedirs('./model/', exist_ok=True)
    torch.save(model.state_dict(), './model/model.pth')

    # Plot the training loss and test loss
    draw_training_and_test_loss()


if __name__ == "__main__":
    main(sys.argv)
