"""
Claire Liu, Yu-Jing Wei
-----
This file contains code to fine-tune a pre-trained model on a new dataset of
Greek letter images. It includes data loading, training, and prediction
functions, as well as visualization of results. The model is modified to
output 3 classes corresponding to the Greek letters alpha, beta, and gamma.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from train_model import MyNetwork

# Define class labels for the Greek letter dataset
LABELS = {0: 'alpha', 1: 'beta', 2: 'gamma'}


class GreekTransform:
    """Transform to preprocess Greek letter images for the model"""
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Define the transformation pipeline for the Greek letter dataset
GREEK_TRANSFORM_PIPELINE = torchvision.transforms.Compose([
    GreekTransform(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])


def train_greek(model, optimizer, greek_train, epoch, losses):
    """Train the model on the Greek letter dataset for one epoch"""
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


def plot_training_loss(losses):
    """Plot the training loss over epochs"""
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('NLL Loss')
    plt.title('Greek Letter Training Loss')
    plt.savefig('greek_training_loss.png')
    plt.show()


def predict_image(img_path, model, labels):
    """Predict the class of a single image using the trained model"""

    img = Image.open(img_path)
    # transform image and unsqueeze it to add batch dimension
    img_tensor = GREEK_TRANSFORM_PIPELINE(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return pred


def predict_greek_images(image_dir, model):
    """Predict the classes of multiple images and return a list
    of modified images and a list of predictions"""
    my_images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    predictions = []
    print("\nResults on custom images:")
    for img_path in my_images:
        if os.path.exists(img_path):
            pred = predict_image(img_path, model, LABELS)
            predictions.append((img_path, pred))
            print(f'{img_path} → Predicted: {LABELS[pred]}')
        else:
            print(f'{img_path} not found.')
    return my_images, predictions


def plot_greek_predictions(my_images, predictions, labels):
    """Plot the predictions of the model on the custom images with
    correct/incorrect coloring"""

    n = len(my_images)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    fig.suptitle("Greek Letter Predictions", fontsize=16)
    axes = axes.flat if n > 1 else [axes]

    for i, (img_path, pred) in enumerate(predictions):
        img = Image.open(img_path)
        # filename as true label (without extension)
        true_label = os.path.splitext(os.path.basename(img_path))[0]

        # Display image
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        # For debugging, you can also display the transformed image:
        # ax.imshow(GREEK_TRANSFORM_PIPELINE(img)[0], cmap='gray')

        # Check if prediction is correct and set title color accordingly
        correct = labels[pred] == true_label
        color = 'green' if correct else 'red'

        ax.set_title(f'Pred: {labels[pred]}, True: {true_label}', color=color)
        ax.axis('off')

    # Hide remaining axes if there are fewer images than subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """Main function to load the model, fine-tune on Greek letter dataset,
    and plot predictions"""

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
            transform=GREEK_TRANSFORM_PIPELINE
        ),
        batch_size=5,
        shuffle=True
    )

    # Print class to index mapping
    print("\nClass to index mapping:")
    print(greek_train.dataset.class_to_idx)

    # Use Adam optimizer to only update the new fc2 layer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    # Train the model for multiple epochs and record losses
    losses = []
    n_epochs = 50
    for epoch in range(1, n_epochs + 1):
        train_greek(network, optimizer, greek_train, epoch, losses)

    # Plot training loss
    plot_training_loss(losses)

    # Predict on custom images and plot results
    images_dir = './data/greek_test/'
    my_images, predictions = predict_greek_images(
        images_dir, network)

    # Plot predictions with correct/incorrect coloring
    plot_greek_predictions(my_images, predictions, LABELS)


if __name__ == "__main__":
    main()
