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
BATCH_SIZE_TEST  = 1000

# ─── Search Space ─────────────────────────────────────────────────────────────
# Phase 1 dimensions (fixed to best found values)
L_options = [0.01]                # Learning Rate       — fixed
M_options = [(64, 128)]           # Conv filters        — fixed
N_options = [0.5]                 # Dropout Rate        — fixed
O_options = ['SGD']               # Optimizer           — fixed
P_options = [20]                  # Epochs              — fixed

# Phase 2 dimensions (new — to explore)
Q_options = [64, 128, 256]        # Hidden nodes in FC layer
R_options = [32, 64, 128]         # Batch size
S_options = ['ReLU', 'LeakyReLU'] # Activation function

# ─── Starting Defaults ────────────────────────────────────────────────────────
DEFAULT_LR           = 0.01
DEFAULT_FILTERS      = (64, 128)
DEFAULT_DROPOUT      = 0.5
DEFAULT_OPTIMIZER    = 'SGD'
DEFAULT_EPOCHS       = 20
DEFAULT_HIDDEN_NODES = 128
DEFAULT_BATCH_SIZE   = 64
DEFAULT_ACTIVATION   = 'ReLU'


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def get_data_loaders(batch_size=DEFAULT_BATCH_SIZE):
    """Load Fashion MNIST train and test sets"""
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            './files/', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.2860,), (0.3530,))
            ])),
        batch_size=batch_size, shuffle=True
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
    hidden_nodes: number of nodes in FC layer
    activation:   'ReLU' or 'LeakyReLU'
    """

    def __init__(self, filters=DEFAULT_FILTERS, dropout_rate=DEFAULT_DROPOUT,
                 hidden_nodes=DEFAULT_HIDDEN_NODES, activation=DEFAULT_ACTIVATION):
        super().__init__()

        f1, f2 = filters

        # Activation function
        self.act = nn.ReLU() if activation == 'ReLU' else nn.LeakyReLU()

        # Conv layers with 3x3 kernel and same padding to preserve spatial size
        self.conv1 = nn.Conv2d(1, f1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)

        # After two MaxPool2d(2): 28 -> 14 -> 7
        # FC input size = f2 * 7 * 7
        self.fc1 = nn.Linear(f2 * 7 * 7, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, 10)

    def forward(self, x):
        # conv1 → act → pool: 28x28 -> 14x14
        x = self.act(F.max_pool2d(self.conv1(x), 2))
        # conv2 → dropout → act → pool: 14x14 -> 7x7
        x = self.act(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Flatten: f2 * 7 * 7
        f2 = x.shape[1]
        x = x.view(-1, f2 * 7 * 7)

        x = self.act(self.fc1(x))
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

def run_experiment(config, test_loader):
    """Train and evaluate one configuration. Returns final test accuracy."""
    # Create train loader with config's batch size
    train_loader, _ = get_data_loaders(batch_size=config['batch_size'])

    model = MyNetwork(
        filters=config['filters'],
        dropout_rate=config['dropout'],
        hidden_nodes=config['hidden_nodes'],
        activation=config['activation']
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

def run_search(test_loader):
    """
    Linear search strategy: hold parameters constant, optimize 1 at a time.
    Phase 1 params are fixed. Explores Q (hidden nodes), R (batch size), S (activation).
    """

    # Fixed from Phase 1
    best_lr        = DEFAULT_LR
    best_filters   = DEFAULT_FILTERS
    best_dropout   = DEFAULT_DROPOUT
    best_optimizer = DEFAULT_OPTIMIZER
    best_epochs    = DEFAULT_EPOCHS

    # Phase 2 — new dimensions to explore
    best_hidden_nodes = DEFAULT_HIDDEN_NODES
    best_batch_size   = DEFAULT_BATCH_SIZE
    best_activation   = DEFAULT_ACTIVATION

    results       = []
    experiment_id = 0

    def base_config():
        """Return current best config as a dict"""
        return {
            'lr':           best_lr,
            'filters':      best_filters,
            'dropout':      best_dropout,
            'optimizer':    best_optimizer,
            'epochs':       best_epochs,
            'hidden_nodes': best_hidden_nodes,
            'batch_size':   best_batch_size,
            'activation':   best_activation,
        }

    def run(config):
        """Run one experiment, log and record the result"""
        nonlocal experiment_id
        experiment_id += 1
        acc = run_experiment(config, test_loader)
        results.append({**config, 'accuracy': acc, 'id': experiment_id})
        print(f"[{experiment_id:02d}] "
              f"hidden={config['hidden_nodes']}  "
              f"batch={config['batch_size']}  "
              f"act={config['activation']}  "
              f"=> {acc:.2f}%")
        return acc

    # Round-robin until ~60 experiments
    round_num = 0
    while experiment_id < 60:
        round_num += 1
        print(f"\n{'='*55}")
        print(f"Round {round_num}")
        print(f"{'='*55}")

        # ── Optimize Q: Hidden Nodes ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Hidden Nodes")
        best_acc = -1
        for hidden_nodes in Q_options:
            acc = run({**base_config(), 'hidden_nodes': hidden_nodes})
            if acc > best_acc:
                best_acc         = acc
                best_hidden_nodes = hidden_nodes
        print(f"  Best hidden_nodes = {best_hidden_nodes}  ({best_acc:.2f}%)")

        # ── Optimize R: Batch Size ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Batch Size")
        best_acc = -1
        for batch_size in R_options:
            acc = run({**base_config(), 'batch_size': batch_size})
            if acc > best_acc:
                best_acc       = acc
                best_batch_size = batch_size
        print(f"  Best batch_size = {best_batch_size}  ({best_acc:.2f}%)")

        # ── Optimize S: Activation ──
        if experiment_id >= 60:
            break
        print("\n→ Optimizing Activation Function")
        best_acc = -1
        for activation in S_options:
            acc = run({**base_config(), 'activation': activation})
            if acc > best_acc:
                best_acc       = acc
                best_activation = activation
        print(f"  Best activation = {best_activation}  ({best_acc:.2f}%)")

    # ── Final Summary ──
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*55}")
    print(f"Search complete!  Total experiments: {experiment_id}")
    print(f"Best config found:")
    print(f"  hidden_nodes = {best_hidden_nodes}")
    print(f"  batch_size   = {best_batch_size}")
    print(f"  activation   = {best_activation}")
    print(f"Best accuracy: {best_result['accuracy']:.2f}%  "
          f"(experiment #{best_result['id']})")

    return results, base_config()


# ─── Save & Plot Results ──────────────────────────────────────────────────────

def save_results(results):
    """Save all experiment results to a CSV file"""
    os.makedirs('./results/', exist_ok=True)
    filepath = './results/search_results_phase2.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['id', 'lr', 'filters', 'dropout', 'optimizer',
                           'epochs', 'hidden_nodes', 'batch_size', 'activation',
                           'accuracy'])
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
    plt.title('Phase 2 Hyperparameter Search: Accuracy per Experiment')
    plt.legend()
    plt.tight_layout()
    os.makedirs('./results/', exist_ok=True)
    plt.savefig('./results/search_results_phase2.png')
    plt.show()
    print("Plot saved to ./results/search_results_phase2.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv):
    _, test_loader = get_data_loaders()

    print("Starting Phase 2 hyperparameter search...")
    print("Fixed from Phase 1:")
    print(f"  lr={DEFAULT_LR}, filters={DEFAULT_FILTERS}, dropout={DEFAULT_DROPOUT}, "
          f"optimizer={DEFAULT_OPTIMIZER}, epochs={DEFAULT_EPOCHS}")
    print("\nExploring new dimensions:")
    print(f"  Q (Hidden Nodes)  : {Q_options}")
    print(f"  R (Batch Size)    : {R_options}")
    print(f"  S (Activation)    : {S_options}")

    results, best_config = run_search(test_loader)

    save_results(results)
    plot_results(results)

    # ── Train final model with best config and save ──
    print(f"\n{'='*55}")
    print("Training final model with best config...")
    print(f"  hidden_nodes = {best_config['hidden_nodes']}")
    print(f"  batch_size   = {best_config['batch_size']}")
    print(f"  activation   = {best_config['activation']}")

    train_loader, _ = get_data_loaders(batch_size=best_config['batch_size'])
    final_model = MyNetwork(
        filters=best_config['filters'],
        dropout_rate=best_config['dropout'],
        hidden_nodes=best_config['hidden_nodes'],
        activation=best_config['activation']
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