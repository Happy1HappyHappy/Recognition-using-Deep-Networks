"""
Claire Liu, Yu-Jing Wei
Examine the trained network 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import cv2

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        pass

def main():
    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()

    # Print model structure (shows layer names and shapes)
    print("=== Model Structure ===")
    print(model)
    print()

    # Get the weights of the first conv layer
    weights = model.conv1.weight

    print("=== conv1 Weight ===")
    print(f"Shape: {weights.shape}")  
    print()

    # Print each filter's weights
    for i in range(weights.shape[0]):
        print(f"Filter {i}:")
        print(weights[i, 0].detach().numpy())
        print()

    # Visualize the 10 filters in a 3x4 grid 
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("conv1 Filters (10 filters, 5x5)")

    for i in range(10):
        plt.subplot(3, 4, i + 1)          
        plt.imshow(weights[i, 0].detach().numpy())
        plt.title(f"Filter {i}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    # Load the first training image
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=True, download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=1, shuffle=False)

    # Get the first image as a numpy array
    first_image, _ = next(iter(train_loader))
    img = first_image[0, 0].numpy()  

    # Apply each filter to the image using cv2.filter2D
    # Visualize filter and filtered image side by side in a 4x5 grid
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Filters and Their Effects on First Training Image")

    with torch.no_grad():
        for i in range(10):
            kernel = weights[i, 0].numpy()
            filtered = cv2.filter2D(img, -1, kernel)

            # Left: the filter itself
            plt.subplot(5, 4, 2 * i + 1)
            plt.imshow(kernel, cmap='gray')
            plt.title(f"Filter {i}")
            plt.xticks([])
            plt.yticks([])

            # Right: result of applying the filter
            plt.subplot(5, 4, 2 * i + 2)
            plt.imshow(filtered, cmap='gray')
            plt.title(f"Filtered {i}")
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()