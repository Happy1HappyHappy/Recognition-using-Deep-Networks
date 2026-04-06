"""
Claire Liu, Yu-Jing Wei
-----
This file tests the trained model on the MNIST test dataset.
It reads the network and runs the model on the first 10 examples
in the test set. It also plots the first 9 examples with their
predicted labels.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
from main import MyNetwork

MODEL_PATH = './model/model.pth'


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


if __name__ == '__main__':
    main()
