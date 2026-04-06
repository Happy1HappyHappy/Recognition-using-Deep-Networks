"""
Claire Liu, Yu-Jing Wei
-----
This file tests the trained model on the MNIST test dataset.
It reads the network and runs the model on the first 10 examples
in the test set. It also plots the first 9 examples with their
predicted labels.
"""

import os
import torch
import torchvision
import matplotlib.pyplot as plt
from train_model import MyNetwork

MODEL_PATH = './model/model.pth'


def test_digits_with_predictions(model, test_loader):
    """Test the trained model on the MNIST test dataset and display results"""

    # Get the first batch of test examples
    examples = enumerate(test_loader)
    _, (data, targets) = next(examples)

    # Run the model on the test examples without computing gradients
    with torch.no_grad():
        output = model(data)

    # Print 10 examples including output values, predicted and correct label
    for i, output_i in enumerate(output[:10]):
        values = output_i.exp()
        pred = output_i.argmax().item()
        print(f'Example {i+1}:')
        print(f'  Output values: {[f"{v:.2f}" for v in values.tolist()]}')
        print(f'  Predicted: {pred}, Correct: {targets[i].item()}')

    # Plot the first 9 examples as a 3x3 grid with predictions
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle('MNIST Test Examples with Predictions', fontsize=16)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        pred = output[i].argmax().item()
        plt.title(f'Prediction: {pred}')
        plt.xticks([])
        plt.yticks([])
    plt.show()


def test_handwritten_digits(model, image_dir='./data/hand_write_digts/'):
    """This function loads handwritten digit images from a directory,
    preprocesses them to match MNIST format, runs them through the network,
    and displays the results."""

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load all images from the directory, sorted by filename
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    images = []
    preds = []

    with torch.no_grad():
        for fname in image_files:
            path = os.path.join(image_dir, fname)
            img = torchvision.io.read_image(path)  # load raw for display
            tensor = transform(
                torchvision.transforms.functional.to_pil_image(img)
            ).unsqueeze(0)  # add batch dimension

            output = model(tensor)
            pred = output.argmax(dim=1).item()
            images.append(img)
            preds.append(pred)

    # Display results
    n = len(images)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle("Handwritten Digit Predictions", fontsize=16)
    axes = axes.flat

    for i, (img, pred) in enumerate(zip(images, preds)):
        ax = axes[i]
        # Convert to grayscale numpy for display
        img_gray = torchvision.transforms.functional.to_pil_image(img).convert('L')
        ax.imshow(img_gray, cmap='gray')
        ax.set_title(f'Pred: {pred}')
        ax.axis('off')

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """Test the trained model on the MNIST test dataset"""

    # Load the test dataset
    test_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(
                        './files/', train=False, download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                        ])),
                    batch_size=100,
                    shuffle=True
                    )

    # Load the model and set it to evaluation mode(disables dropout)
    model = MyNetwork()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Test on MNIST test dataset and display results
    test_digits_with_predictions(model, test_loader)

    # Test on handwritten digits
    test_handwritten_digits(model)


if __name__ == '__main__':
    main()
