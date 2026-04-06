"""
Claire Liu, Yu-Jing Wei
Transfer Learning on Greek Letters
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from train_model import MyNetwork


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def train_greek(model, optimizer, greek_train, epoch, losses):
    model.train()
    epoch_loss = 0

    for data, target in greek_train:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(greek_train)
    losses.append(avg_loss)
    print(f'Epoch {epoch}: Loss={avg_loss:.4f}')

# Predict a single image
def predict_image(img_path, model, labels):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    print(f'{img_path} → Predicted: {labels[pred]}')
    return pred


def main():
    # Load pre-trained model
    network = MyNetwork()
    network.load_state_dict(torch.load('./model/model.pth'))
    print("Original network:")
    print(network)

    # Freeze weights of all layers 
    for param in network.parameters():
        param.requires_grad = False

    # Replace the final Fully Connected layer to output 3 classes 
    network.fc2 = nn.Linear(50, 3)
    print("\nModified network:")
    print(network)

    # DataLoader
    training_set_path = './data/greek_train/' 

    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )

    # Print class to index mapping
    print("\nClass to index mapping:")
    print(greek_train.dataset.class_to_idx)

    # Use Adam optimizer to only update the new fc2 layer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    losses = []
    n_epochs = 50

    for epoch in range(1, n_epochs + 1):
        train_greek(network, optimizer, greek_train, epoch, losses)

    # Plot training loss
    plt.figure()
    plt.plot(range(1, n_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('NLL Loss')
    plt.title('Greek Letter Training Loss')
    plt.savefig('greek_training_loss.png')
    plt.show()

    labels = {0: 'alpha', 1: 'beta', 2: 'gamma'}

    my_images = [
        './data/greek_test/alpha.jpg',
        './data/greek_test/beta.jpg',
        './data/greek_test/gamma.jpg',
    ]

    print("\nResults on custom images:")
    for img_path in my_images:
        if os.path.exists(img_path):
            predict_image(img_path, network, labels)
        else:
            print(f'{img_path} not found.')


if __name__ == "__main__":
    main()