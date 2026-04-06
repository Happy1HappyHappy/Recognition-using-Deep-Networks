import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from train_model import MyNetwork  

def main():
    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()

    # Define the transform to match MNIST preprocessing
    # Invert to black background and white digits
    transform = transforms.Compose([
        transforms.Grayscale(),           
        transforms.Resize((28, 28)),      
        transforms.ToTensor(),           
        transforms.Lambda(lambda x: 1 - x), 
        transforms.Normalize((0.1307,), (0.3081,))  
    ])

    # Load each digit image
    images = []
    for i in range(10):
        img = Image.open(f'./data/digit_{i}.png')
        img_tensor = transform(img)
        images.append(img_tensor)

    # Stack into a batch
    batch = torch.stack(images)

    # Run through the network
    with torch.no_grad():
        output = model(batch)

    # Print results
    for i in range(10):
        pred = output[i].argmax().item()
        values = output[i].exp()
        print(f'Digit {i}:')
        print(f'  Output values: {[f"{v:.2f}" for v in values.tolist()]}')
        print(f'  Predicted: {pred}, Correct label: {i}')

    # Plot all 10 digits with predictions
    fig = plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap='gray', interpolation='none')
        pred = output[i].argmax().item()
        plt.title(f'Pred: {pred}\nTrue: {i}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()