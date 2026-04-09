# Recognition using Deep Networks

## Authors
- Claire Liu
- Yu-Jing Wei

## Overview
This project focuses on building, training, analyzing, and modifying a deep network for a digit recognition task. The models are trained using the PyTorch framework on the MNIST dataset, which contains 60,000 training and 10,000 testing images of 28x28 grayscale handwritten digits. The project progresses from basic CNN training to network analysis, transfer learning, and exploring vision transformers.

## Directory Structure
```text
.
├── data/                  # Custom input images (e.g., hand-written digits and Greek letters)
├── files/                 # PyTorch datasets (downloaded automatically)
├── model/                 # Saved model weights
│   └── model.pth          # Saved CNN model (generated after training)
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── train_model.py         # Building and training the CNN
├── test_model.py          # Evaluate on MNIST test set & custom digits
├── examine_network.py     # Print network structure & visualize conv1 filters
├── greek_letters.py       # Transfer learning for Greek symbols
├── transformer.py         # Custom Vision Transformer network
└── img_preprocessor.py    # Helper utility for custom image preprocessing
```

## Project Structure
- `train_model.py`: Defines the CNN architecture (`MyNetwork`), downloads the MNIST dataset, and trains the model for 5 epochs. The trained model weights are saved to `./model/model.pth`. Also plots training and testing loss.
- `test_model.py`: Loads the trained model and evaluates it on the MNIST test set. Prints network predictions and plots the first 9 test examples with predicted labels. It also tests the model on custom hand-drawn digit images ([0-9]), preprocessing them to match the MNIST style before evaluation.
- `examine_network.py`: Examines the internal structure of the model, specifically analyzing and visualizing the weights of the first convolution layer (`conv1`). Applies these filters to a training example to visualize the filtering effect.
- `greek_letters.py`: Implements transfer learning to classify Greek letters (alpha, beta, gamma) using the pre-trained MNIST CNN network. Freezes early layer weights and swaps the final classification layer.
- `transformer.py`: Re-implements the digit recognition network utilizing transformer layers in place of the convolutional layers.
- `img_preprocessor.py`: Utilities for custom image preprocessing.
- `data/` and `files/`: Directories containing the default PyTorch dataset files and custom handwritten images/Greek letters.
- `model/`: Directory storing the trained model states (e.g., `model.pth`).

## Environment Setup
It is recommended to use PyTorch, TorchVision, and OpenCV.
Install the dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Tasks and Execution

### Build and Train a Network
Create and compile the CNN with specified layers (10 5x5 filters Convolution, MaxPooling, 20 5x5 filters Convolution, Dropout, Linear). Run the following to train:
```bash
python train_model.py
```
This generates the `./model/model.pth` file and visualizes the training & testing loss curves.

### Evaluate the Network on MNIST Test Set & New Handwritten Digits
To evaluate the network performance, examine predictions on the MNIST test set, and test it on custom handwritten digits:
```bash
python test_model.py
```
It sets the model to evaluation mode (`network.eval()`), evaluates test set accuracy, prints the top predictions for the first 10 examples, and displays a 3x3 grid of the first 9 digits. Downstream, it reads custom handwritten digit images, preprocesses them to match the MNIST format, and evaluates the model performance on these novel inputs (ensure your custom digits are thick and inverted).

### Examine the Network Architecture & Filters
Print the network structure and visualize the 10 `conv1` layer 5x5 filters alongside their effects on a sample image:
```bash
python examine_network.py
```

### Transfer Learning on Greek Letters
Replace the output layer of the MNIST model, freeze earlier weights, and train to classify custom Greek symbols (alpha, beta, gamma). Uses `GreekTransform` for custom resizing and inversion:
```bash
python greek_letters.py
```

Link to hand-written Greek letters: https://drive.google.com/file/d/12OF7sIhxc5B9pP5RHRPD1luPmyVSo2Ko/view?usp=sharing

### Transformer Network
A complete replacement of the CNN layers with a Vision Transformer-like design utilizing patching, tokenizer, and transformer encoder blocks:
```bash
python transformer.py
```
