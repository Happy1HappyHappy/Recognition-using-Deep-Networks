"""
Claire Liu, Yu-Jing Wei
-----
Building a Convolutional Neural Network with PyTorch.
Fashion MNIST with automated hyperparameter search using linear search strategy.
"""

import os
import sys
import csv
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# ─── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')       # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device('mps')        # Apple Silicon GPU
else:
    device = torch.device('cpu')        # CPU fallback
print(f"Using device: {device}")

# ─── Data Hyperparameters ─────────────────────────────────────────────────────
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST  = 1000

# ─── Search Space ─────────────────────────────────────────────────────────────
L_options = [0.001, 0.005, 0.01]             # Learning Rate
M_options = [(16, 32), (32, 64), (64, 128)]  # (conv1 filters, conv2 filters)
N_options = [0.25, 0.35, 0.5]               # Dropout Rate
O_options = ['SGD', 'Adam']                  # Optimizer
P_options = [10, 15, 20]                     # Number of Epochs

# ─── Starting Defaults ────────────────────────────────────────────────────────
DEFAULT_LR        = 0.01
DEFAULT_FILTERS   = (32, 64)
DEFAULT_DROPOUT   = 0.35
DEFAULT_OPTIMIZER = 'SGD'
DEFAULT_EPOCHS    = 10


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def get_data_loaders():
    """Load Fashion MNIST train and test sets"""
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            './files/', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.2860,), (0.3530,))
            ])),
        batch_size=BATCH_SIZE_TRAIN, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            './files/', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.2860,), (0.3530,))
            ])),
        batch_size=BATCH_SIZE_TEST, shuffle=False
    )
    return train_loader, test_loader


# ─── Network ──────────────────────────────────────────────────────────────────

class MyNetwork(nn.Module):
    """
    Configurable CNN for Fashion MNIST classification.
    filters:      tuple of (conv1_filters, conv2_filters)
    dropout_rate: dropout probability after conv2
    """

    def __init__(self, filters=DEFAULT_FILTERS, dropout_rate=DEFAULT_DROPOUT):
        super().__init__()

        f1, f2 = filters

        # Conv layers with 3x3 kernel and same padding to preserve spatial size
        self.conv1 = nn.Conv2d(1, f1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)

        # After two MaxPool2d(2): 28 -> 14 -> 7
        # FC input size = f2 * 7 * 7
        self.fc1 = nn.Linear(f2 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # conv1 → ReLU → pool: 28x28 -> 14x14
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2 → dropout → ReLU → pool: 14x14 -> 7x7
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Flatten: f2 * 7 * 7
        f2 = x.shape[1]
        x = x.view(-1, f2 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ─── Train / Test ─────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, train_loader):
    """Train the model for one epoch"""
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader):
    """Evaluate the model on the test set, return (accuracy, avg_loss)"""
    model.eval()
    correct   = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output     = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred       = output.argmax(dim=1)
            correct   += pred.eq(target).sum().item()
    test_loss /= len(test_loader)
    accuracy   = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss


# ─── Run One Experiment ───────────────────────────────────────────────────────

def run_experiment(config, train_loader, test_loader):
    """Train and evaluate one configuration. Returns final test accuracy."""
    model = MyNetwork(
        filters=config['filters'],
        dropout_rate=config['dropout']
    ).to(device)

    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config['lr'], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['lr'])

    for _ in range(config['epochs']):
        train_epoch(model, optimizer, train_loader)

    accuracy, _ = evaluate(model, test_loader)
    return accuracy


# ─── Linear Search ────────────────────────────────────────────────────────────

def run_search(train_loader, test_loader):
    """
    Linear search strategy: hold 4 parameters constant, optimize 1 at a time.
    Cycles round-robin through all 5 dimensions until ~60 experiments are done.
    """

    # Current best values, initialized to defaults
    best_lr        = DEFAULT_LR
    best_filters   = DEFAULT_FILTERS
    best_dropout   = DEFAULT_DROPOUT
    best_optimizer = DEFAULT_OPTIMIZER
    best_epochs    = DEFAULT_EPOCHS

    results       = []
    experiment_id = 0

    def base_config():
        """Return current best config as a dict"""
        return {
            'lr':        best_lr,
            'filters':   best_filters,
            'dropout':   best_dropout,
            'optimizer': best_optimizer,
            'epochs':    best_epochs
        }

    def run(config):
        """Run one experiment, log and record the result"""
        nonlocal experiment_id
        experiment_id += 1
        acc = run_experiment(config, train_loader, test_loader)
        results.append({**config, 'accuracy': acc, 'id': experiment_id})
        print(f"[{experiment_id:02d}] lr={config['lr']}  "
              f"filters={config['filters']}  "
              f"dropout={config['dropout']}  "
              f"opt={config['optimizer']}  "
              f"epochs={config['epochs']}  "
              f"=> {acc:.2f}%")
        return acc

    # Round-robin until ~60 experiments
    round_num = 0
    while experiment_id < 60:
        round_num += 1
        print(f"\n{'='*55}")
        print(f"Round {round_num}")
        print(f"{'='*55}")

        # ── Optimize L: Learning Rate ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Learning Rate")
        best_acc = -1
        for lr in L_options:
            acc = run({**base_config(), 'lr': lr})
            if acc > best_acc:
                best_acc = acc
                best_lr  = lr
        print(f"  Best lr = {best_lr}  ({best_acc:.2f}%)")

        # ── Optimize M: Filters ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Filters")
        best_acc = -1
        for filters in M_options:
            acc = run({**base_config(), 'filters': filters})
            if acc > best_acc:
                best_acc     = acc
                best_filters = filters
        print(f"  Best filters = {best_filters}  ({best_acc:.2f}%)")

        # ── Optimize N: Dropout ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Dropout")
        best_acc = -1
        for dropout in N_options:
            acc = run({**base_config(), 'dropout': dropout})
            if acc > best_acc:
                best_acc     = acc
                best_dropout = dropout
        print(f"  Best dropout = {best_dropout}  ({best_acc:.2f}%)")

        # ── Optimize O: Optimizer ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Optimizer")
        best_acc = -1
        for opt in O_options:
            acc = run({**base_config(), 'optimizer': opt})
            if acc > best_acc:
                best_acc       = acc
                best_optimizer = opt
        print(f"  Best optimizer = {best_optimizer}  ({best_acc:.2f}%)")

        # ── Optimize P: Epochs ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Epochs")
        best_acc = -1
        for epochs in P_options:
            acc = run({**base_config(), 'epochs': epochs})
            if acc > best_acc:
                best_acc    = acc
                best_epochs = epochs
        print(f"  Best epochs = {best_epochs}  ({best_acc:.2f}%)")

    # ── Final Summary ──
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*55}")
    print(f"Search complete!  Total experiments: {experiment_id}")
    print(f"Best config found:")
    print(f"  lr        = {best_lr}")
    print(f"  filters   = {best_filters}")
    print(f"  dropout   = {best_dropout}")
    print(f"  optimizer = {best_optimizer}")
    print(f"  epochs    = {best_epochs}")
    print(f"Best accuracy: {best_result['accuracy']:.2f}%  "
          f"(experiment #{best_result['id']})")

    return results, base_config()


# ─── Save & Plot Results ──────────────────────────────────────────────────────

def save_results(results):
    """Save all experiment results to a CSV file"""
    os.makedirs('./results/', exist_ok=True)
    filepath = './results/search_results.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['id', 'lr', 'filters', 'dropout',
                           'optimizer', 'epochs', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {filepath}")


def plot_results(results):
    """Plot accuracy across all experiments"""
    ids  = [r['id'] for r in results]
    accs = [r['accuracy'] for r in results]

    plt.figure(figsize=(12, 5))
    plt.plot(ids, accs, 'o-', color='steelblue', markersize=4)
    plt.axhline(y=max(accs), color='red', linestyle='--',
                label=f'Best: {max(accs):.2f}%')
    plt.xlabel('Experiment #')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Hyperparameter Search: Accuracy per Experiment')
    plt.legend()
    plt.tight_layout()
    os.makedirs('./results/', exist_ok=True)
    plt.savefig('./results/search_results.png')
    plt.show()
    print("Plot saved to ./results/search_results.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv):
    train_loader, test_loader = get_data_loaders()

    print("Starting hyperparameter search...")
    print("Dimensions:")
    print(f"  L (Learning Rate) : {L_options}")
    print(f"  M (Filters)       : {M_options}")
    print(f"  N (Dropout)       : {N_options}")
    print(f"  O (Optimizer)     : {O_options}")
    print(f"  P (Epochs)        : {P_options}")
    print(f"\nDefaults: lr={DEFAULT_LR}, filters={DEFAULT_FILTERS}, "
          f"dropout={DEFAULT_DROPOUT}, optimizer={DEFAULT_OPTIMIZER}, "
          f"epochs={DEFAULT_EPOCHS}")

    results, best_config = run_search(train_loader, test_loader)

    save_results(results)
    plot_results(results)

    # ── Train final model with best config and save ──
    print(f"\n{'='*55}")
    print("Training final model with best config...")
    print(f"  lr        = {best_config['lr']}")
    print(f"  filters   = {best_config['filters']}")
    print(f"  dropout   = {best_config['dropout']}")
    print(f"  optimizer = {best_config['optimizer']}")
    print(f"  epochs    = {best_config['epochs']}")

    # Train final model with best config
    final_model = MyNetwork(
        filters=best_config['filters'],
        dropout_rate=best_config['dropout']
    ).to(device)
    if best_config['optimizer'] == 'SGD':
        final_optimizer = torch.optim.SGD(
            final_model.parameters(), lr=best_config['lr'], momentum=0.9)
    else:
        final_optimizer = torch.optim.Adam(
            final_model.parameters(), lr=best_config['lr'])

    for epoch in range(1, best_config['epochs'] + 1):
        train_epoch(final_model, final_optimizer, train_loader)
        print(f"  Epoch {epoch}/{best_config['epochs']} done")

    final_acc, _ = evaluate(final_model, test_loader)
    print(f"\nFinal model accuracy: {final_acc:.2f}%")

    os.makedirs('./model/', exist_ok=True)
    torch.save(final_model.state_dict(), './model/best_model.pth')
    print("Final model saved to ./model/best_model.pth")


if __name__ == "__main__":
    main(sys.argv)