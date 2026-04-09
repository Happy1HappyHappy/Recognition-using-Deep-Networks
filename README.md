# Recognition using Deep Networks

## Authors
- Claire Liu
- Yu-Jing Wei

## Overview
This project focuses on building, training, analyzing, and modifying deep neural networks for image classification tasks using the PyTorch framework. Initially utilizing the MNIST dataset (handwritten digits), the project progresses from basic CNN training and network analysis to advanced transfer learning on 6 custom Greek letters (evaluated with confusion matrices and t-SNE feature space visualization). Furthermore, the repository explores Vision Transformer-like architectures and expands to the Fashion MNIST dataset, implementing custom CNNs alongside automated linear hyperparameter search strategies to methodically optimize model performance.

## Directory Structure
```text
.
├── data/                  # Custom input images (e.g., hand-written digits and Greek letters)
├── files/                 # PyTorch datasets (downloaded automatically)
├── model/                 # Saved model weights
│   └── model.pth          # Saved CNN model (generated after training)
├── results/               # Experiment results, plots, and CSVs
├── venv/                  # Python virtual environment (ignored in git)
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── train_model.py         # Building and training the CNN
├── train_model_gabor.py   # Training CNN initialized with fixed Gabor filters
├── test_model.py          # Evaluate on MNIST test set & custom digits
├── examine_network.py     # Print network structure & visualize conv1 filters
├── greek_letters.py       # Transfer learning for Greek symbols
├── transformer.py         # Custom Vision Transformer network
├── img_preprocessor.py    # Helper utility for custom image preprocessing
├── owned_network.py       # Custom CNN implementation for Fashion MNIST
└── owned_network_opt.py   # Automated hyperparameter search for the custom CNN
```

## Project Structure
- `train_model.py`: Defines the CNN architecture (`MyNetwork`), downloads the MNIST dataset, and trains the model for 5 epochs. The trained model weights are saved to `./model/model.pth`. Also plots training and testing loss.
- `train_model_gabor.py`: Initializes the first convolutional layer with fixed Gabor filters using OpenCV, turning off gradients for this layer. Trains the remaining layers on MNIST and saves the model as `./model/model_gabor.pth`.
- `test_model.py`: Loads the trained model and evaluates it on the MNIST test set. Prints network predictions and plots the first 9 test examples with predicted labels. It also tests the model on custom hand-drawn digit images ([0-9]), preprocessing them to match the MNIST style before evaluation.
- `examine_network.py`: Examines the internal structure of the model, specifically analyzing and visualizing the weights of the first convolution layer (`conv1`). Applies these filters to a training example to visualize the filtering effect.
- `greek_letters.py`: Implements transfer learning to classify 6 Greek letters (alpha, beta, delta, epsilon, gamma, theta) using the pre-trained MNIST CNN network. Freezes early layer weights and swaps the final classification layer. Includes confusion matrix and t-SNE feature space visualization.
- `transformer.py`: Re-implements the digit recognition network utilizing transformer layers in place of the convolutional layers.
- `img_preprocessor.py`: Utilities for custom image preprocessing.
- `owned_network.py`: Implements a simple CNN to classify the Fashion MNIST dataset, including functions to display samples and loss curves.
- `owned_network_opt.py`: Expands on `owned_network.py` by conducting an automated hyperparameter search using a linear search strategy. Trains final network with best configs and saves results.
- `data/` and `files/`: Directories containing the default PyTorch dataset files and custom handwritten images/Greek letters.
- `model/`: Directory storing the trained model states (e.g., `model.pth`).
- `results/`: Directory storing outputs from hyperparameter search, such as `search_results.csv` and evaluation plots.

## Environment Setup
It is recommended to use PyTorch, TorchVision, and OpenCV.
First, create and activate a Python virtual environment to manage dependencies locally:
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

Next, install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Tasks and Execution

### Build and Train a Network
Create and compile the CNN with specified layers (10 5x5 filters Convolution, MaxPooling, 20 5x5 filters Convolution, Dropout, Linear). Run the following to train:
```bash
python3 train_model.py
```
This generates the `./model/model.pth` file and visualizes the training & testing loss curves.

### Train a Network with Initialized Gabor Filters
Initialize the first convolution layer with 10 Gabor filters of different orientations, freezing their weights, and train the rest of the CNN on the MNIST dataset:
```bash
python3 train_model_gabor.py
```
This generates the `./model/model_gabor.pth` file and plots training & testing loss curves.

### Evaluate the Network on MNIST Test Set & New Handwritten Digits
To evaluate the network performance, examine predictions on the MNIST test set, and test it on custom handwritten digits:
```bash
python3 test_model.py
```
It sets the model to evaluation mode (`network.eval()`), evaluates test set accuracy, prints the top predictions for the first 10 examples, and displays a 3x3 grid of the first 9 digits. Downstream, it reads custom handwritten digit images, preprocesses them to match the MNIST format, and evaluates the model performance on these novel inputs (ensure your custom digits are thick and inverted).

### Examine the Network Architecture & Filters
Print the network structure and visualize the 10 `conv1` layer 5x5 filters alongside their effects on a sample image:
```bash
python3 examine_network.py
```

### Transfer Learning on Greek Letters
Replace the output layer of the MNIST model, freeze earlier weights, and train to classify 6 custom Greek symbols (alpha, beta, delta, epsilon, gamma, theta). Uses `GreekTransform` for custom resizing and inversion, and evaluates performance using a confusion matrix and t-SNE visualization of the 50D feature space:
```bash
python3 greek_letters.py
```

Link to hand-written Greek letters: https://drive.google.com/file/d/12OF7sIhxc5B9pP5RHRPD1luPmyVSo2Ko/view?usp=sharing

### Transformer Network
A complete replacement of the CNN layers with a Vision Transformer-like design utilizing patching, tokenizer, and transformer encoder blocks:
```bash
python3 transformer.py
```

### Fashion MNIST Custom Network
Run the custom CNN implementation for the Fashion MNIST classification dataset to compile and evaluate the model:
```bash
python3 owned_network.py
```

### Fashion MNIST Hyperparameter Search
Run the linear hyperparameter search (learning rate, filter sizes, dropout, optimizer, epochs) to find the best configuration, train the final model, and save search results:
```bash
python3 owned_network_opt.py
```
