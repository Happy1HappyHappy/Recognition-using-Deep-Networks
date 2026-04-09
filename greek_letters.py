"""
Claire Liu, Yu-Jing Wei
-----
This file contains code to fine-tune a pre-trained model on a new dataset of
Greek letter images. It includes data loading, training, and prediction
functions, as well as visualization of results. The model is modified to
output 3 classes corresponding to the Greek letters alpha, beta, and gamma.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE  # for Dimension Reduction
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
    plt.show()


def predict_greek_images(test_loader, model):
    """Predict the classes of multiple images and return a list
    of modified images and a list of predictions"""
    model.eval()
    images_list = []
    preds_list = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()
            for i in range(data.shape[0]):
                images_list.append(data[i][0].cpu().numpy())
                preds_list.append(preds[i])

    return images_list, preds_list


def plot_greek_predictions(my_images, predictions, labels, test_dataset):
    """Plot the predictions of the model on the custom images with
    correct/incorrect coloring"""

    n = len(my_images)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    fig.suptitle("Greek Letter Predictions", fontsize=16)
    axes = axes.flat if n > 1 else [axes]

    for i in range(n):
        # Get the image data and predicted index for the current image
        img_data = my_images[i]
        pred_idx = predictions[i]

        # Get the true label index from the test dataset samples
        _, true_idx = test_dataset.samples[i]
        true_label = labels[true_idx]
        pred_label = labels[pred_idx]

        ax = axes[i]
        # Normalize the image data for better visualization
        img_display = (
            (img_data - img_data.min()) / (img_data.max() - img_data.min()))
        ax.imshow(img_display, cmap='gray')

        # Determine if the prediction is correct and set title color
        correct = (pred_idx == true_idx)
        color = 'green' if correct else 'red'

        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}',
                     color=color, fontsize=10)
        ax.axis('off')

    # Hide remaining axes if there are fewer images than subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_extra_dimensions(model, dataloader, labels_dict):
    """Evaluate the model on the dataset and plot additional dimensions
    such as confusion matrix and t-SNE visualization of features"""
    model.eval()
    all_preds, all_targets, all_features = [], [], []

    with torch.no_grad():
        for data, target in dataloader:
            # Extract the 50-dimensional feature vector from fc1 output
            x = model.conv1(data)
            x = F.max_pool2d(F.relu(x), 2)
            x = model.conv2(x)
            x = F.max_pool2d(F.relu(x), 2)
            x = x.view(-1, 320)
            x = F.relu(model.fc1(x))
            all_features.append(x.cpu().numpy())

            # Get the predicted class from the final output layer
            output = model.fc2(x)
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.array(all_targets)
    label_names = [labels_dict[i] for i in range(len(labels_dict))]

    # --- Confusion Matrix ---
    # Plot the confusion matrix to evaluate the model's performance on
    # each class and identify any misclassifications
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix(all_targets, all_preds), annot=True, fmt='d',
                cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

    # --- t-SNE Visualization ---
    # Use t-SNE to reduce the 50D feature space to 2D for visualization
    # Cluster the features based on their true labels to see if the model
    # has learned meaningful representations

    # Set a low perplexity to better capture local structure
    # in the small dataset
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    features_2d = tsne.fit_transform(all_features)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(label_names):
        mask = all_targets == i
        plt.scatter(
            features_2d[mask, 0], features_2d[mask, 1], label=name, s=100)
    plt.title('t-SNE Feature Space (50D -> 2D)')
    plt.legend()
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
        shuffle=False
    )

    # Print class to index mapping
    print("\nClass to index mapping:")
    print(greek_train.dataset.class_to_idx)

    # Use Adam optimizer to only update the new fc2 layer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

    # Train the model for multiple epochs and record losses
    losses = []
    n_epochs = 45
    for epoch in range(1, n_epochs + 1):
        train_greek(network, optimizer, greek_train, epoch, losses)

    # Plot training loss
    plot_training_loss(losses)

    # Predict on custom images and plot results
    images_dir = './data/greek_test/'
    test_dataset = torchvision.datasets.ImageFolder(
        root=images_dir,
        transform=GREEK_TRANSFORM_PIPELINE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False)

    my_images, predictions = predict_greek_images(test_loader, network)

    # Plot predictions with correct/incorrect coloring
    plot_greek_predictions(my_images, predictions, LABELS, test_dataset)

    # Evaluate extra dimensions: confusion matrix and t-SNE visualization
    eval_loader = torch.utils.data.DataLoader(
        greek_train.dataset, batch_size=len(greek_train.dataset))
    evaluate_extra_dimensions(network, eval_loader, LABELS)


if __name__ == "__main__":
    main()
