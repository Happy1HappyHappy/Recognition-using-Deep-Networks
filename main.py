"""
Claire Liu, Yu-Jing Wei

"""

import os
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Define model
class MyNetwork(nn.Module):
    # Setup the layers of the network in the constructor
    def __init__(self):
        super(MyNetwork, self).__init__()
        # A convolution layer with 1 channel(greyscale) and 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        # A dropout layer with a 0.5 dropout rate (50%)
        self.conv2_drop = nn.Dropout2d(p=0.5)

        # Fully connected Linear layer with 50 nodes
        # 24x24 -> 20x20 -> 10x10 -> 4x4
        # 20 channels * 4 * 4 = 320
        self.fc1 = nn.Linear(320, 50)
        
        # Fully connected Linear layer with 50 nodes
        # To identify 10 digits, we need 10 output nodes
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        # conv1 → pooling → ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2 → dropout → pooling → ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Flattern output, 20 channels * 4 * 4 = 320
        x = x.view(-1, 320)
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # Fully connected Linear layer with 10 nodes 
        x = self.fc2(x)
        # log_softmax function applied to the output
        return F.log_softmax(x, dim=1)
    

# useful functions with a comment for each function
def train_network(model, optimizer, train_loader, epoch, train_losses, train_counter, batch_size):
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Clear the gradients
        optimizer.zero_grad()          
        # perform a forward pass
        output = model(data)          
        # compute the loss
        loss = F.nll_loss(output, target)  
        # perform a backward pass
        loss.backward()                
        # update the weights
        optimizer.step()     

        # Calculate accuracy
        train_losses.append(loss.item())
        # Record the index of the current batch for plotting
        train_counter.append(
            (batch_idx * batch_size) + ((epoch - 1) * 60000)
        )         

def test_network(model, test_loader, test_losses):
    model.eval()
    correct = 0
    test_loss = 0
    
    # For testing, we don't calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    # Calculate accuracy
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'Test Accuracy = {accuracy:.2f}%')

def main():
    # Hyperparameters

    # Sections of training sets
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    # Training sets and load it into DataSet
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=False)

    # Check the data, iterate and print
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    # Plot the first 6 images in the test set and their corresponding labels
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    
    plt.show()


    # Create an instance of the network
    model = MyNetwork()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum)
    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    # Train the network and test it after each epoch
    for epoch in range(1, n_epochs + 1):
        train_network(model, optimizer, train_loader, epoch,
                      train_losses, train_counter, batch_size_train)
        test_network(model, test_loader, test_losses)
        test_counter.append(epoch * 60000)
    
    # Save the model
    os.makedirs('./model/', exist_ok=True)
    torch.save(model.state_dict(), './model/model.pth')

    # Plot the training loss and test loss
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue', label='Train loss')
    plt.scatter(test_counter, test_losses, color='red', zorder=5, label='Test loss')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()



        

