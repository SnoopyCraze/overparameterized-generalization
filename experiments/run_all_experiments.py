"""
Experiments for: Implicit Regularization and Generalization in Overparameterized Neural Networks
Optimized for RTX 3070 Ti (8GB). Target runtime: ~30-40 minutes total.
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NUM_SEEDS = 2

print(f"Device: {DEVICE}", flush=True)
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


# ============================================================
# DATA
# ============================================================

def load_cifar10():
    t = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    tr = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=t)
    te = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=t)
    trx = torch.stack([tr[i][0] for i in range(len(tr))])
    try_ = torch.tensor([tr[i][1] for i in range(len(tr))])
    tex = torch.stack([te[i][0] for i in range(len(te))])
    tey = torch.tensor([te[i][1] for i in range(len(te))])
    return trx, try_, tex, tey

def load_mnist():
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    tr = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=t)
    te = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=t)
    trx = torch.stack([tr[i][0] for i in range(len(tr))]).view(-1, 784)
    try_ = torch.tensor([tr[i][1] for i in range(len(tr))])
    tex = torch.stack([te[i][0] for i in range(len(te))]).view(-1, 784)
    tey = torch.tensor([te[i][1] for i in range(len(te))])
    return trx, try_, tex, tey


# ============================================================
# MODELS
# ============================================================

class MLP(nn.Module):
    def __init__(self, width, depth=4, input_dim=784, num_classes=10):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

class SimpleCNN(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(3,c,3,padding=1), nn.ReLU(), nn.Conv2d(c,c,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c,2*c,3,padding=1), nn.ReLU(), nn.Conv2d(2*c,2*c,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2*c,4*c,3,padding=1), nn.ReLU(), nn.Conv2d(4*c,4*c,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(4*c*4*4, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.classifier(self.features(x))
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# TRAINING
# ============================================================

def train_model(model, trx, try_, tex, tey, epochs, bs=128, lr=0.1,
                opt_type='sgd', wd=0, verbose_interval=0):
    model = model.to(DEVICE)
    trx_d, try_d = trx.to(DEVICE), try_.to(DEVICE)
    tex_d, tey_d = tex.to(DEVICE), tey.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = trx_d.size(0)
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        correct, total, rloss = 0, 0, 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            x, y = trx_d[idx], try_d[idx]
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            rloss += loss.item() * x.size(0)
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)

        train_acc = 100.*correct/total
        train_loss = rloss/total
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out = model(tex_d)
            test_loss = criterion(out, tey_d).item()
            test_acc = 100.*(out.argmax(1)==tey_d).sum().item()/tey_d.size(0)

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        if verbose_interval and (epoch+1) % verbose_interval == 0:
            print(f"    epoch {epoch+1}/{epochs}: train={train_acc:.1f}% test={test_acc:.1f}%", flush=True)

    return history


def train_model_with_mask(model, mask, trx, try_, tex, tey, epochs, bs=128, lr=0.1):
    """Train with weight mask applied (for lottery ticket experiments)."""
    model = model.to(DEVICE)
    trx_d, try_d = trx.to(DEVICE), try_.to(DEVICE)
    tex_d, tey_d = tex.to(DEVICE), tey.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = trx_d.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            x, y = trx_d[idx], try_d[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in mask and p.grad is not None:
                        p.grad.mul_(mask[name])
            optimizer.step()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in mask:
                        p.mul_(mask[name])
        scheduler.step()

    model.eval()
    with torch.no_grad():
        test_acc = 100.*(model(tex_d).argmax(1)==tey_d).sum().item()/tey_d.size(0)
    return test_acc


# ============================================================
# EXPERIMENT 1: DOUBLE DESCENT
# ============================================================

def experiment_double_descent(mnist_data, cifar_data):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 1: Double Descent", flush=True)
    print("="*60, flush=True)
    mtrx, mtry, mtex, mtey = mnist_data
    ctrx, ctry, ctex, ctey = cifar_data

    results = {'mlp_mnist': {}, 'cnn_cifar10': {}}

    # MLP on MNIST (fast — 784-dim input, 60k samples)
    # Use 10k subset to make interpolation threshold accessible
    n_sub = 10000
    torch.manual_seed(0)
    idx = torch.randperm(mtrx.size(0))[:n_sub]
    sub_trx, sub_try = mtrx[idx], mtry[idx]

    mlp_widths = [8, 16, 32, 64, 128, 256, 512, 1024]
    print(f"\n--- MLP on MNIST (n={n_sub}) ---", flush=True)

    for w in mlp_widths:
        sd = {'params': None, 'train_acc': [], 'test_acc': []}
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            model = MLP(width=w, depth=4, input_dim=784)
            sd['params'] = model.count_params()
            h = train_model(model, sub_trx, sub_try, mtex, mtey,
                           epochs=60, bs=128, lr=0.05)
            sd['train_acc'].append(h['train_acc'][-1])
            sd['test_acc'].append(h['test_acc'][-1])
            del model; torch.cuda.empty_cache()

        mt = np.mean(sd['train_acc']); me = np.mean(sd['test_acc']); se = np.std(sd['test_acc'])
        print(f"  w={w:>5}: params={sd['params']:>10,} | train={mt:.1f}% test={me:.1f}+/-{se:.1f}%", flush=True)
        results['mlp_mnist'][str(w)] = sd

    # CNN on CIFAR-10 (use full 50k)
    cnn_channels = [4, 8, 16, 32, 64, 96]
    print(f"\n--- CNN on CIFAR-10 (n=50000) ---", flush=True)

    for ch in cnn_channels:
        sd = {'params': None, 'train_acc': [], 'test_acc': []}
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            model = SimpleCNN(base_channels=ch)
            sd['params'] = model.count_params()
            h = train_model(model, ctrx, ctry, ctex, ctey,
                           epochs=60, bs=128, lr=0.1)
            sd['train_acc'].append(h['train_acc'][-1])
            sd['test_acc'].append(h['test_acc'][-1])
            del model; torch.cuda.empty_cache()

        mt = np.mean(sd['train_acc']); me = np.mean(sd['test_acc']); se = np.std(sd['test_acc'])
        print(f"  ch={ch:>3}: params={sd['params']:>10,} | train={mt:.1f}% test={me:.1f}+/-{se:.1f}%", flush=True)
        results['cnn_cifar10'][str(ch)] = sd

    with open(os.path.join(RESULTS_DIR, 'exp1_double_descent.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (key, title, n_train) in enumerate([
        ('mlp_mnist', 'MLP on MNIST', n_sub),
        ('cnn_cifar10', 'CNN on CIFAR-10', 50000)
    ]):
        params, test_m, test_s, train_m = [], [], [], []
        for k in results[key]:
            r = results[key][k]
            params.append(r['params'])
            test_m.append(np.mean(r['test_acc']))
            test_s.append(np.std(r['test_acc']))
            train_m.append(np.mean(r['train_acc']))
        ax = axes[idx]
        ax.errorbar(params, [100-t for t in test_m], yerr=test_s, fmt='o-', label='Test Error', capsize=3, linewidth=2)
        ax.plot(params, [100-t for t in train_m], 's--', label='Train Error', alpha=0.7)
        ax.axvline(x=n_train, color='gray', ls=':', alpha=0.7, label=f'n={n_train:,} (train size)')
        ax.set_xscale('log')
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Error (%)')
        ax.set_title(f'Double Descent — {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig1_double_descent.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_double_descent.png", flush=True)
    return results


# ============================================================
# EXPERIMENT 2: IMPLICIT REGULARIZATION
# ============================================================

def experiment_implicit_reg(ctrx, ctry, ctex, ctey):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 2: Implicit Regularization", flush=True)
    print("="*60, flush=True)

    configs = [
        ('SGD_bs64',   'sgd',  64,   0.1),
        ('SGD_bs128',  'sgd',  128,  0.1),
        ('SGD_bs512',  'sgd',  512,  0.1),
        ('SGD_bs2048', 'sgd',  2048, 0.1),
        ('Adam_bs128', 'adam', 128,  0.001),
    ]
    epochs = 60
    results = {}

    for name, opt, bs, lr in configs:
        print(f"  {name}...", end=' ', flush=True)
        sd = {'train_acc': [], 'test_acc': [], 'gen_gap': []}
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            model = SimpleCNN(base_channels=64)
            h = train_model(model, ctrx, ctry, ctex, ctey,
                           epochs=epochs, bs=bs, lr=lr, opt_type=opt, wd=0)
            ft, fe = h['train_acc'][-1], h['test_acc'][-1]
            sd['train_acc'].append(ft); sd['test_acc'].append(fe); sd['gen_gap'].append(ft-fe)
            del model; torch.cuda.empty_cache()
        mt = np.mean(sd['test_acc']); st = np.std(sd['test_acc']); mg = np.mean(sd['gen_gap'])
        print(f"test={mt:.2f}+/-{st:.2f}%, gap={mg:.2f}%", flush=True)
        results[name] = sd

    # Full-batch GD on 5k subset
    print("  FullBatch_GD...", end=' ', flush=True)
    sd = {'train_acc': [], 'test_acc': [], 'gen_gap': []}
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        idx = torch.randperm(ctrx.size(0))[:5000]
        model = SimpleCNN(base_channels=64)
        h = train_model(model, ctrx[idx], ctry[idx], ctex, ctey,
                       epochs=epochs, bs=5000, lr=0.01, opt_type='sgd', wd=0)
        ft, fe = h['train_acc'][-1], h['test_acc'][-1]
        sd['train_acc'].append(ft); sd['test_acc'].append(fe); sd['gen_gap'].append(ft-fe)
        del model; torch.cuda.empty_cache()
    mt = np.mean(sd['test_acc']); st = np.std(sd['test_acc']); mg = np.mean(sd['gen_gap'])
    print(f"test={mt:.2f}+/-{st:.2f}%, gap={mg:.2f}%", flush=True)
    results['FullBatch_GD'] = sd

    with open(os.path.join(RESULTS_DIR, 'exp2_implicit_reg.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    tm = [np.mean(results[n]['test_acc']) for n in names]
    ts = [np.std(results[n]['test_acc']) for n in names]
    gg = [np.mean(results[n]['gen_gap']) for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, tm, w, yerr=ts, label='Test Accuracy (%)', capsize=3)
    ax.bar(x+w/2, gg, w, label='Generalization Gap (%)', color='orange')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('%'); ax.set_title('Implicit Regularization: Optimizer & Batch Size')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig2_implicit_reg.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_implicit_reg.png", flush=True)
    return results


# ============================================================
# EXPERIMENT 3: LOSS LANDSCAPE
# ============================================================

def experiment_loss_landscape(ctrx, ctry, ctex, ctey):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 3: Loss Landscape", flush=True)
    print("="*60, flush=True)

    criterion = nn.CrossEntropyLoss()
    eval_x, eval_y = ctrx[:5000].to(DEVICE), ctry[:5000].to(DEVICE)

    configs = {'small_batch_64': 64, 'large_batch_2048': 2048}
    trained = {}

    for name, bs in configs.items():
        print(f"  Training {name}...", end=' ', flush=True)
        torch.manual_seed(42)
        model = SimpleCNN(base_channels=64)
        h = train_model(model, ctrx, ctry, ctex, ctey, epochs=60, bs=bs, lr=0.1, wd=0)
        print(f"test={h['test_acc'][-1]:.2f}%", flush=True)
        trained[name] = model

    # Perturbation analysis
    print("\n  Perturbation analysis...", flush=True)
    sigmas = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    pert_results = {}

    for name, model in trained.items():
        model.eval()
        with torch.no_grad():
            base_loss = criterion(model(eval_x), eval_y).item()

        data = []
        for sigma in sigmas:
            losses = []
            for _ in range(5):
                pm = copy.deepcopy(model)
                with torch.no_grad():
                    for p in pm.parameters():
                        p.add_(torch.randn_like(p) * sigma)
                    pl = criterion(pm(eval_x), eval_y).item()
                losses.append(pl)
                del pm
            mean_pl = np.mean(losses)
            pct = ((mean_pl - base_loss) / (base_loss + 1e-8)) * 100
            data.append({'sigma': sigma, 'base_loss': float(base_loss),
                        'perturbed_loss': float(mean_pl), 'pct_increase': float(pct)})
        pert_results[name] = data
        print(f"    {name}: base={base_loss:.4f}, sigma=0.01 -> +{data[4]['pct_increase']:.1f}%", flush=True)

    # Hessian top eigenvalue via power iteration
    print("\n  Hessian eigenvalue estimation...", flush=True)
    hess_results = {}
    hess_x, hess_y = ctrx[:1000].to(DEVICE), ctry[:1000].to(DEVICE)

    for name, model in trained.items():
        print(f"    {name}...", end=' ', flush=True)
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]

        v = [torch.randn_like(p) for p in params]
        v_norm = torch.sqrt(sum((vi**2).sum() for vi in v))
        v = [vi / v_norm for vi in v]

        top_eig = 0
        for _ in range(25):
            model.zero_grad()
            loss = criterion(model(hess_x), hess_y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            gv = sum((g * vi).sum() for g, vi in zip(grads, v))
            hvp = torch.autograd.grad(gv, params)
            hv = [h.detach() for h in hvp]
            top_eig = sum((vi * hi).sum().item() for vi, hi in zip(v, hv))
            hv_norm = torch.sqrt(sum((hi**2).sum() for hi in hv))
            v = [hi / (hv_norm + 1e-10) for hi in hv]

        hess_results[name] = {'top_eigenvalue': float(top_eig)}
        print(f"top_eig = {top_eig:.2f}", flush=True)

    landscape = {'perturbation': pert_results, 'hessian': hess_results}
    with open(os.path.join(RESULTS_DIR, 'exp3_loss_landscape.json'), 'w') as f:
        json.dump(landscape, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in pert_results.items():
        axes[0].plot([d['sigma'] for d in data], [d['pct_increase'] for d in data],
                     'o-', label=name, linewidth=2)
    axes[0].set_xlabel('Perturbation Magnitude (sigma)')
    axes[0].set_ylabel('Loss Increase (%)')
    axes[0].set_title('Loss Sensitivity to Weight Perturbation')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    names_h = list(hess_results.keys())
    eigs = [hess_results[n]['top_eigenvalue'] for n in names_h]
    axes[1].bar(names_h, eigs, color=['steelblue', 'coral'])
    axes[1].set_ylabel('Top Hessian Eigenvalue')
    axes[1].set_title('Loss Landscape Curvature (Sharpness)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig3_loss_landscape.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_loss_landscape.png", flush=True)

    for m in trained.values(): del m
    torch.cuda.empty_cache()
    return landscape


# ============================================================
# EXPERIMENT 4: LOTTERY TICKET
# ============================================================

def experiment_lottery_ticket(ctrx, ctry, ctex, ctey):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 4: Lottery Ticket Hypothesis", flush=True)
    print("="*60, flush=True)

    epochs = 50
    prune_pcts = [0, 50, 70, 80, 90, 95]
    results = {}

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        model = SimpleCNN(base_channels=64).to(DEVICE)
        init_state = copy.deepcopy(model.state_dict())

        h = train_model(model, ctrx, ctry, ctex, ctey, epochs=epochs, bs=128, lr=0.1, wd=0)
        trained_state = copy.deepcopy(model.state_dict())

        for pp in prune_pcts:
            key = f"prune_{pp}pct"
            if key not in results:
                results[key] = {'prune_pct': pp, 'test_acc': []}

            if pp == 0:
                results[key]['test_acc'].append(h['test_acc'][-1])
                print(f"  seed={seed} prune={pp}%: test={h['test_acc'][-1]:.2f}%", flush=True)
                continue

            # Magnitude pruning mask
            wkeys = [k for k in trained_state if 'weight' in k and trained_state[k].dim() >= 2]
            all_w = torch.cat([trained_state[k].abs().flatten() for k in wkeys])
            thresh = torch.quantile(all_w.float(), pp / 100.0)
            mask = {k: (trained_state[k].abs() >= thresh).float().to(DEVICE) for k in wkeys}

            pruned = SimpleCNN(base_channels=64).to(DEVICE)
            pruned.load_state_dict(init_state)
            with torch.no_grad():
                sd = pruned.state_dict()
                for k in mask:
                    sd[k].mul_(mask[k])

            test_acc = train_model_with_mask(pruned, mask, ctrx, ctry, ctex, ctey,
                                             epochs=epochs, bs=128, lr=0.1)
            results[key]['test_acc'].append(test_acc)
            print(f"  seed={seed} prune={pp}%: test={test_acc:.2f}%", flush=True)
            del pruned; torch.cuda.empty_cache()

        del model; torch.cuda.empty_cache()

    # Also test random reinit (control)
    print("  Random reinit control (90% pruned)...", end=' ', flush=True)
    results['random_reinit_90pct'] = {'prune_pct': 90, 'test_acc': []}
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        model = SimpleCNN(base_channels=64).to(DEVICE)
        h = train_model(model, ctrx, ctry, ctex, ctey, epochs=epochs, bs=128, lr=0.1, wd=0)
        trained_state = model.state_dict()

        wkeys = [k for k in trained_state if 'weight' in k and trained_state[k].dim() >= 2]
        all_w = torch.cat([trained_state[k].abs().flatten() for k in wkeys])
        thresh = torch.quantile(all_w.float(), 0.9)
        mask = {k: (trained_state[k].abs() >= thresh).float().to(DEVICE) for k in wkeys}

        # Random reinit instead of original init
        torch.manual_seed(seed + 100)
        pruned = SimpleCNN(base_channels=64).to(DEVICE)
        with torch.no_grad():
            sd = pruned.state_dict()
            for k in mask: sd[k].mul_(mask[k])

        test_acc = train_model_with_mask(pruned, mask, ctrx, ctry, ctex, ctey,
                                         epochs=epochs, bs=128, lr=0.1)
        results['random_reinit_90pct']['test_acc'].append(test_acc)
        del model, pruned; torch.cuda.empty_cache()
    print(f"test={np.mean(results['random_reinit_90pct']['test_acc']):.2f}%", flush=True)

    with open(os.path.join(RESULTS_DIR, 'exp4_lottery_ticket.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # Main pruning curve
    pp_vals, tm_vals, ts_vals = [], [], []
    for key in sorted([k for k in results if k != 'random_reinit_90pct'],
                      key=lambda k: results[k]['prune_pct']):
        r = results[key]
        pp_vals.append(100 - r['prune_pct'])
        tm_vals.append(np.mean(r['test_acc']))
        ts_vals.append(np.std(r['test_acc']))

    ax.errorbar(pp_vals, tm_vals, yerr=ts_vals, fmt='o-', capsize=3, linewidth=2,
                label='Original init (Lottery Ticket)')

    # Random reinit point
    ri = results['random_reinit_90pct']
    ax.errorbar([10], [np.mean(ri['test_acc'])], yerr=[np.std(ri['test_acc'])],
                fmt='x', markersize=10, capsize=3, color='red', linewidth=2,
                label='Random reinit (90% pruned)')

    ax.set_xlabel('Remaining Parameters (%)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Lottery Ticket: Pruning vs Performance')
    ax.invert_xaxis()
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig4_lottery_ticket.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_lottery_ticket.png", flush=True)
    return results


# ============================================================
# EXPERIMENT 5: NTK REGIME
# ============================================================

def experiment_ntk_regime(mtrx, mtry, mtex, mtey):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 5: NTK Regime", flush=True)
    print("="*60, flush=True)

    # 5k subset of MNIST
    sub_trx, sub_try = mtrx[:5000], mtry[:5000]
    widths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    epochs = 30
    results = {}

    for w in widths:
        sd = {'width': w, 'params': None, 'param_movement': [], 'test_acc': [], 'loss_var': []}
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            model = MLP(width=w, depth=4, input_dim=784).to(DEVICE)
            sd['params'] = model.count_params()

            if sd['params'] * 4 > 3e9:
                print(f"  w={w}: SKIP (too large)", flush=True)
                break

            init_params = {n: p.clone().detach() for n, p in model.named_parameters()}
            h = train_model(model, sub_trx, sub_try, mtex, mtey,
                           epochs=epochs, bs=128, lr=0.01)

            diff_sq = sum(((p.detach()-init_params[n])**2).sum().item() for n,p in model.named_parameters())
            init_sq = sum((init_params[n]**2).sum().item() for n in init_params)
            rel_mov = np.sqrt(diff_sq) / (np.sqrt(init_sq) + 1e-8)

            sd['param_movement'].append(float(rel_mov))
            sd['test_acc'].append(h['test_acc'][-1])
            sd['loss_var'].append(float(np.var(h['train_loss'][-10:])))
            del model; torch.cuda.empty_cache()

        if sd['test_acc']:
            mt = np.mean(sd['test_acc']); mm = np.mean(sd['param_movement'])
            print(f"  w={w:>5}: params={sd['params']:>10,} | test={mt:.1f}% | movement={mm:.4f}", flush=True)
            results[str(w)] = sd

    with open(os.path.join(RESULTS_DIR, 'exp5_ntk_regime.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ws = [results[k]['width'] for k in results]
    mm = [np.mean(results[k]['param_movement']) for k in results]
    ms = [np.std(results[k]['param_movement']) for k in results]
    tm = [np.mean(results[k]['test_acc']) for k in results]
    ts_ = [np.std(results[k]['test_acc']) for k in results]

    axes[0].errorbar(ws, mm, yerr=ms, fmt='o-', capsize=3, linewidth=2)
    axes[0].set_xlabel('Network Width'); axes[0].set_ylabel('Relative Parameter Movement')
    axes[0].set_title('NTK: Parameter Movement vs Width')
    axes[0].set_xscale('log'); axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(ws, tm, yerr=ts_, fmt='s-', capsize=3, linewidth=2, color='green')
    axes[1].set_xlabel('Network Width'); axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('NTK: Test Accuracy vs Width')
    axes[1].set_xscale('log'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig5_ntk_regime.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_ntk_regime.png", flush=True)
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    start = time.time()
    print("="*60, flush=True)
    print("RUNNING ALL EXPERIMENTS", flush=True)
    print("="*60, flush=True)

    print("\nLoading datasets...", flush=True)
    t0 = time.time()
    mnist_data = load_mnist()
    cifar_data = load_cifar10()
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    experiment_double_descent(mnist_data, cifar_data)
    ctrx, ctry, ctex, ctey = cifar_data
    experiment_implicit_reg(ctrx, ctry, ctex, ctey)
    experiment_loss_landscape(ctrx, ctry, ctex, ctey)
    experiment_lottery_ticket(ctrx, ctry, ctex, ctey)

    del cifar_data, ctrx, ctry, ctex, ctey
    torch.cuda.empty_cache()

    mtrx, mtry, mtex, mtey = mnist_data
    experiment_ntk_regime(mtrx, mtry, mtex, mtey)

    elapsed = time.time() - start
    print(f"\n{'='*60}", flush=True)
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes", flush=True)
    print(f"Results in: {RESULTS_DIR}", flush=True)
    print(f"{'='*60}", flush=True)
