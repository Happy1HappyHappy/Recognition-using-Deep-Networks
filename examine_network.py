"""
Claire Liu, Yu-Jing Wei
-----
Examine the trained network and visualize the
filters in the first convolutional layer.
"""

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import cv2

MODEL_PATH = './model/model.pth'


class MyNetwork(nn.Module):
    """MyNetwork is the same architecture as the one used for training.
   We need to define it here to load the saved model parameters"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
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


def main():
    """This function loads the trained model, prints the weights of the
    first convolutional layer, and visualizes the filters and their effects
    on the first training image"""

    # Load the trained model and set it to evaluation mode
    model = MyNetwork()
    model.load_state_dict(torch.load(MODEL_PATH))
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
    weights_size = weights.shape[0]
    for i in range(weights_size):
        print(f"Filter {i}:")
        # detach the weights from the computation graph and convert
        # to numpy for printing
        print(weights[i, 0].detach().numpy())
        print()

    # Visualize the 10 filters in a 3x4 grid
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("conv1 Filters (10 filters, 5x5)")

    for i in range(weights_size):
        plt.subplot(3, 4, i + 1)
        plt.imshow(weights[i, 0].detach().numpy())
        plt.title(f"Filter {i}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    # Load the first training image
    train_loader = torch.utils.data.DataLoader(
                        torchvision.datasets.MNIST(
                            './files/', train=True, download=False,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
                        batch_size=1,
                        shuffle=False
                        )

    # iter() creates an iterator from the dataloader, and next() gets the
    # first batch (which is just one image here)
    first_image, _ = next(iter(train_loader))
    # Convert tensor to numpy array for filtering
    # [0, 0] selects the first image and the first channel (grayscale)
    img = first_image[0, 0].numpy()

    # Apply each filter to the image using cv2.filter2D
    # Visualize filter and filtered image side by side in a 4x5 grid
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Filters and Their Effects on First Training Image")

    with torch.no_grad():
        for i in range(weights_size):
            kernel = weights[i, 0].numpy()
            # Apply the filter to the image using OpenCV's filter2D function
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
