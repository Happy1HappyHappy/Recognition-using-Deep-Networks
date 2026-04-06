"""
Claire Liu, Yu-Jing Wei
-----
This file implements a Vision Transformer (ViT) architecture for
image classification tasks.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


class NetConfig:
    """Configuration class for the Vision Transformer model, containing
    hyperparameters and dataset-specific settings."""
    def __init__(self,
                 name='vit_base',
                 dataset='mnist',
                 patch_size=4,
                 stride=2,
                 embed_dim=48,
                 depth=4,
                 num_heads=8,
                 mlp_dim=128,
                 dropout=0.1,
                 use_cls_token=False,
                 epochs=15,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-4,
                 seed=0,
                 optimizer='adamw',
                 device='mps',
                 ):

        # data set fixed attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # variable things
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,\
                Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%0.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name,
            self.dataset,
            self.patch_size,
            self.stride,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.mlp_dim,
            self.dropout,
            self.use_cls_token,
            self.epochs,
            self.batch_size,
            self.lr,
            self.weight_decay,
            self.seed,
            self.optimizer
        )
        self.config_string = s
        return


class PatchEmbedding(nn.Module):
    """
    A Vision Transformer splits the image into small patches, then turns
    each patch into a token embedding.

    Converts an image into a sequence of patch embeddings.
    Input:
        x of shape (B, C, H, W)

    Output:
        tokens of shape (B, N, D)

    where:
        B = batch size
        N = number of patches (tokens)
        D = embedding dimension
    """

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            stride: int,
            in_channels: int,
            embed_dim: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # - non-overlapping patches  (stride == patch_size)
        # - overlapping patches      (stride < patch_size)
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=stride,
        )
        # Each extracted patch is flattened into one vector
        self.patch_dim = in_channels * patch_size * patch_size

        # After flattening a patch, project it into embedding space.
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Precompute how many patches will be produced for this image setup
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self) -> int:
        """
        Compute how many patches are extracted in total.

        Number of positions along one spatial dimension:
            ((image_size - patch_size) // stride) + 1

        Since the image is square and the patch is square, total patches are:
            positions_per_dim * positions_per_dim
        """
        positions_per_dim = (
            (self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches and convert them to embeddings.

        Input:
            x shape = (B, C, H, W)

        Output:
            x shape = (B, N, D)
        """
        # Step 1: extract patches using nn.Unfold, the shape becomes (B, patch_dim, N)
        #   patch_dim = flattened size of one patch
        #   N = number of extracted patches
        x = self.unfold(x)

        # Step 2: move dimensions so each patch becomes one row/token.
        # Shape becomes: (B, N, patch_dim)
        x = x.transpose(1, 2)

        # Step 3: project each flattened patch into embedding space.
        # Shape becomes: (B, N, embed_dim)
        x = self.proj(x)

        return x


class NetTransformer(nn.Module):
    """
    network structure:
        Patch embedding layer
        dropout
        Transformer layer (with dropout)
        Transformer layer (with dropout)
        Transformer layer (with dropout)
        Token averaging
        Linear layer w/GELU and dropout
        Fully connected output layer 10 nodes: softmax output
    """

    def __init__(self, config):
        """the init method defines the layers of the network"""
        # create all of the layers that have to store information
        super().__init__()

        # make the patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # how many tokens are there?
        num_tokens = self.patch_embed.num_patches
        print(f"Number of tokens: {num_tokens}")

        # does it use a classifier token or a global average token?
        self.use_cls_token = config.use_cls_token

        # if it uses a classifier node, create a source for the node
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens+1
        else:  # no CLS token
            self.cls_token = None
            total_tokens = num_tokens

        # need to include a learned positional embedding, one for each token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)  # do I need this?

        # Use the Torch Transformer Encoder Layer
        # transformer layer includes
        # multi-head self attention
        # feedforward network
        # layer normalization
        # residual connections
        # dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.embed_dim,
            nhead = config.num_heads,
            dim_feedforward = config.mlp_dim,
            dropout = config.dropout,
            activation = 'gelu',
            batch_first = True,
            norm_first = True,
        )

        # Create a stack of transformer layers to build an encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = config.depth,
        )

        # final normalization layer prior to classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            # nn.Dropout(config.dropout),  # optional
            nn.Linear(config.mlp_dim, config.num_classes)
        )

        return

    def _init_parameters(self) -> None:
        """
        initialize special parameters
        - positional embedding
        - optional CLS token
        """
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Execute a forward pass through the network
        Input x: (B, 1, 28, 28)
        Output: logits: (B, num_classes)
        """
        # Patch embedding: (B, 1, 28, 28) -> (B, N, embed_dim)
        x = self.patch_embed(x)

        # Get batch size
        batch_size = x.size(0)

        # Prepend CLS token if used
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            # concatenate CLS token to the beginning of the token sequence
            x = torch.cat([cls_token, x], dim=1)

        # Add learnable positional embedding to each token
        x = x + self.pos_embed

        # Apply dropout layer after positional embedding
        x = self.pos_dropout(x)

        # Run transformer encoder
        x = self.encoder(x)

        # Pool tokens into a single vector
        if self.use_cls_token:
            x = x[:, 0]        # use CLS token
        else:
            x = x.mean(dim=1)  # global average pooling

        # Final layer normalization before classification
        x = self.norm(x)

        # Classification MLP
        x = self.classifier(x)

        # Return log softmax
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    """Train the model for one epoch"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target tensors to the configured device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # Print training status every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} \
                [{batch_idx * len(data):5d}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t\
                    Loss: {loss.item():.6f}')


def test(model, device, test_loader):
    """Evaluate the model on the test dataset and print
    average loss and accuracy"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Move data and target tensors to the configured device
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Get the index of the max log-probability as the predicted class
            pred = output.argmax(dim=1, keepdim=True)
            # Count correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, \
        Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


def recognize(model, device, test_loader):
    """Run a recognition demo on a few test samples,
    showing predicted vs true labels"""

    print("\nRunning recognition demo on a few test samples...")

    model.eval()
    # Get one batch of test data
    data, target = next(iter(test_loader))
    # Move data and target tensors to the configured device
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

    # Plot the first 6 images with their predicted and true labels
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(
        "Recognition Demo: Green = Correct, Red = Incorrect", fontsize=16)

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        # Move the image back to CPU, remove batch/channel dimensions,
        # do greyscale and not interpolation for plotting
        plt.imshow(data[i].cpu().squeeze(), cmap='gray', interpolation='none')
        # Set title color to green if prediction is correct, red if incorrect
        plt.title(
            f"Pred: {pred[i].item()}, True: {target[i].item()}",
            color=("green" if pred[i].item() == target[i].item() else "red"))

        plt.xticks([])
        plt.yticks([])
    plt.show()


def main():
    """Main function to set up data, model, training loop, and evaluation"""
    # Initialize configuration
    config = NetConfig()

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # Set up device (GPU if available, otherwise CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Define data transformations: convert to tensor and normalize
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset with the defined transformations
    train_dataset = torchvision.datasets.MNIST(
        './files/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)

    # Load test dataset with the same transformations
    test_dataset = torchvision.datasets.MNIST(
        './files/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize the Vision Transformer model and
    # move it to the configured device
    model = NetTransformer(config).to(device)
    # Initialize special parameters like positional embeddings and CLS token
    # for better training performance
    model._init_parameters()

    # Set up the optimizer based on the configuration (AdamW or Adam)
    if config.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Create a directory to save the model if it doesn't exist
    os.makedirs('./model/', exist_ok=True)

    # Training loop: train for the specified number of epochs,
    # evaluating on the test set after each epoch
    for epoch in range(1, config.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save the trained model's state dictionary to a file for later use
    torch.save(model.state_dict(), './model/transformer_mnist.pth')
    print("Model saved to './model/transformer_mnist.pth'")

    # Run a recognition demo on a few test samples to visualize predictions
    recognize(model, device, test_loader)


if __name__ == "__main__":
    main()
