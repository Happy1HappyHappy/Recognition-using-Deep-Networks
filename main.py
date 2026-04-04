"""
Claire Liu, Yu-Jing Wei

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Define model
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # A convolution layer with 1 channel(greyscale) and 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        # A dropout layer with a 0.5 dropout rate (50%)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        
        # 6. 全連接層 (Linear)：50 個節點
        # 計算方式：經過兩次 5x5 conv 和兩次 2x2 pool 後，28x28 的圖片會變成 4x4
        # 20 個通道 * 4 * 4 = 320
        self.fc1 = nn.Linear(320, 50)
        
        # 7. 最終全連接層：10 個節點 (對應數字 0-9)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # A flattening operation followed by a fully connected Linear layer 
        # with 50 nodes and a ReLU function on the output
        # Flattening
        x = x.view(-1, 320)
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

# useful functions with a comment for each function
def train_network( arguments ):
    return


def main():
    # Hyperparameters

    # Sections of training sets
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1

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

    # Plot the data
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    
    plt.show()

if __name__ == "__main__":
    main()



        

