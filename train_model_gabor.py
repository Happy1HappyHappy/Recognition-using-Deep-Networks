"""
Claire Liu, Yu-Jing Wei
-----
In this file, we define a convolutional neural network with Gabor
filters as the first layer. We then train and test the network on
the MNIST dataset, and plot the training and test losses over time.
"""

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Constants / Hyperparameters
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5

# Lists to store training and test losses, and the number of training examples
train_losses = []
train_counter = []
test_losses = []
test_counter = []


class MyNetworkGabor(nn.Module):
    """A convolutional neural network with Gabor filters as the first layer."""
    def __init__(self):
        super().__init__()

        # first convolutional layer: 1 input channel (grayscale),
        # 10 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # initialize the weights of conv1 with Gabor filters
        gabor_weights = create_gabor_kernels(10, 5)
        with torch.no_grad():
            self.conv1.weight.copy_(gabor_weights)

        # freeze the weights and biases of conv1 so they won't be updated
        # during training
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False

        # second convolutional layer: 10 input channels, 20 output channels,
        # 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)

        # fully connected layers
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """This function defines the forward pass of the network"""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def create_gabor_kernels(n_filters=10, k_size=5):
    """This function creates a set of Gabor filters with different
    orientations, and returns them as a PyTorch Tensor"""
    kernels = []
    # Create Gabor filters with different orientations
    for i in range(n_filters):
        theta = i / n_filters * np.pi  # orientation of the Gabor filter
        # getGaborKernel can be used to create a Gabor filter
        kernel = cv2.getGaborKernel((k_size, k_size), sigma=1.5,
                                    theta=theta, lambd=4.0, gamma=0.5,
                                    psi=0, ktype=cv2.CV_32F)
        kernels.append(kernel)

    # Transform the list of kernels into a PyTorch Tensor with shape
    # (n_filters, 1, k_size, k_size) to be used as the weights of conv1
    return torch.tensor(np.array(kernels)).unsqueeze(1)


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

    # Create an instance of the network
    model = MyNetworkGabor()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                model.parameters()),
                                lr=LEARNING_RATE, momentum=MOMENTUM)

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
    torch.save(model.state_dict(), './model/model_gabor.pth')

    # Plot the training loss and test loss
    draw_training_and_test_loss()


if __name__ == "__main__":
    main(sys.argv)
