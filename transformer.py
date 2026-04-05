import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


class NetConfig:
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

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
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
    Converts an image into a sequence of patch embeddings.
    Input:  x of shape (B, C, H, W)
    Output: tokens of shape (B, N, D)
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

        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=stride,
        )

        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self) -> int:
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, patch_dim, N)
        x = self.unfold(x)
        # (B, patch_dim, N) -> (B, N, patch_dim)
        x = x.transpose(1, 2)
        # (B, N, patch_dim) -> (B, N, embed_dim)
        x = self.proj(x)
        return x


class NetTransformer(nn.Module):

    def __init__(self, config):
        super(NetTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        num_tokens = self.patch_embed.num_patches
        print("Number of tokens: %d" % (num_tokens))

        self.use_cls_token = config.use_cls_token

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens = num_tokens

        self.pos_embed = nn.Parameter(
            torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
        )

        self.norm = nn.LayerNorm(config.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes)
        )
        return

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # 1. Patch embedding: (B, 1, 28, 28) -> (B, N, embed_dim)
        x = self.patch_embed(x)

        # 2. Get batch size
        batch_size = x.size(0)

        # 3. Prepend CLS token if used
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # 4. Add learnable positional embedding
        x = x + self.pos_embed

        # 5. Apply dropout after positional embedding
        x = self.pos_dropout(x)

        # 6. Run transformer encoder
        x = self.encoder(x)

        # 7. Pool tokens into a single vector
        if self.use_cls_token:
            x = x[:, 0]        # use CLS token
        else:
            x = x.mean(dim=1)  # global average pooling

        # 8. Final layer norm
        x = self.norm(x)

        # 9. Classification MLP
        x = self.classifier(x)

        # 10. Return log softmax
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

def recognize(model, device, test_loader):
    model.eval()
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
    
    fig = plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(data[i].cpu().squeeze(), cmap='gray', interpolation='none')
        # 若預測正確則顯示綠色，錯誤為紅色
        plt.title(f"Pred: {pred[i].item()}, True: {target[i].item()}", color=("green" if pred[i].item() == target[i].item() else "red"))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main():
    config = NetConfig()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST('./files/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./files/', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = NetTransformer(config).to(device)
    model._init_parameters()
    
    if config.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    import os
    os.makedirs('./model/', exist_ok=True)
    
    for epoch in range(1, config.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        
    torch.save(model.state_dict(), './model/transformer_mnist.pth')

    print("\nRunning recognition demo on a few test samples...")
    recognize(model, device, test_loader)

if __name__ == "__main__":
    main()