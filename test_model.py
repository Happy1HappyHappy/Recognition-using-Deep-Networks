import torch
import torchvision
import matplotlib.pyplot as plt
from main import MyNetwork

def main():
    # Load the test dataset
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=10, shuffle=False)

    # Load the model and set to evaluation mode
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()

    # Get the first 10 examples from the test set
    examples = enumerate(test_loader)
    batch_idx, (data, targets) = next(examples)

    # Run the model on the first 10 examples
    with torch.no_grad():
        output = model(data)

    # Print the output values, predicted label, and correct label for each example
    for i in range(10):
        values = output[i].exp()
        pred = output[i].argmax().item()
        print(f'Example {i+1}:')
        print(f'  Output values: {[f"{v:.2f}" for v in values.tolist()]}')
        print(f'  Predicted: {pred}, Correct: {targets[i].item()}')

    # Plot the first 9 examples as a 3x3 grid with predictions
    fig = plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        pred = output[i].argmax().item()
        plt.title(f'Prediction: {pred}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()