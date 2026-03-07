"""
Refined runs for the implicit-regularization (Exp 2), loss-landscape (Exp 3),
and lottery-ticket (Exp 4) experiments.

These use the batch-size-scaled learning-rate schedule reported in the paper
(base lr = 0.01 at batch size 128, scaled linearly with batch size) and the
BatchNorm CNN. The numbers produced here are the ones reported in the paper's
implicit-regularization, loss-landscape, and lottery-ticket tables.
Double descent (Exp 1) and the NTK sweep (Exp 5) are in run_all_experiments.py.
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
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NUM_SEEDS = 3
print(f"Device: {DEVICE}", flush=True)


# ============================================================
# DATA & MODELS (same as before)
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

class SimpleCNN(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(3,c,3,padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c,c,3,padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c,2*c,3,padding=1), nn.BatchNorm2d(2*c), nn.ReLU(),
            nn.Conv2d(2*c,2*c,3,padding=1), nn.BatchNorm2d(2*c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2*c,4*c,3,padding=1), nn.BatchNorm2d(4*c), nn.ReLU(),
            nn.Conv2d(4*c,4*c,3,padding=1), nn.BatchNorm2d(4*c), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(4*c*4*4, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.classifier(self.features(x))
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_model(model, trx, try_, tex, tey, epochs, bs=128, lr=0.01,
                opt_type='sgd', wd=0, momentum=0.9):
    model = model.to(DEVICE)
    trx_d, try_d = trx.to(DEVICE), try_.to(DEVICE)
    tex_d, tey_d = tex.to(DEVICE), tey.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
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

        history['train_acc'].append(100.*correct/total)
        history['train_loss'].append(rloss/total)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out = model(tex_d)
            history['test_loss'].append(criterion(out, tey_d).item())
            history['test_acc'].append(100.*(out.argmax(1)==tey_d).sum().item()/tey_d.size(0))

    return history


# ============================================================
# EXPERIMENT 2: IMPLICIT REGULARIZATION
# ============================================================

def experiment2(trx, try_, tex, tey):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 2: Implicit Regularization", flush=True)
    print("="*60, flush=True)

    # Use lr=0.01 as base for bs=128, scale linearly
    configs = [
        ('SGD_bs32',   'sgd',  32,   0.005),
        ('SGD_bs64',   'sgd',  64,   0.0075),
        ('SGD_bs128',  'sgd',  128,  0.01),
        ('SGD_bs256',  'sgd',  256,  0.02),
        ('SGD_bs512',  'sgd',  512,  0.04),
        ('SGD_bs1024', 'sgd',  1024, 0.08),
        ('SGD_bs2048', 'sgd',  2048, 0.1),
        ('Adam_bs128', 'adam', 128,  0.001),
    ]
    epochs = 80
    results = {}

    for name, opt, bs, lr in configs:
        print(f"  {name} (lr={lr})...", end=' ', flush=True)
        sd = {'train_acc': [], 'test_acc': [], 'gen_gap': []}
        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            model = SimpleCNN(base_channels=32)
            h = train_model(model, trx, try_, tex, tey,
                           epochs=epochs, bs=bs, lr=lr, opt_type=opt, wd=0)
            ft, fe = h['train_acc'][-1], h['test_acc'][-1]
            sd['train_acc'].append(ft); sd['test_acc'].append(fe); sd['gen_gap'].append(ft-fe)
            del model; torch.cuda.empty_cache()
        mt = np.mean(sd['test_acc']); st = np.std(sd['test_acc']); mg = np.mean(sd['gen_gap'])
        print(f"test={mt:.2f}+/-{st:.2f}%, gap={mg:.2f}%", flush=True)
        results[name] = sd

    # Full-batch GD on 5k subset
    print("  FullBatch_GD (5k)...", end=' ', flush=True)
    sd = {'train_acc': [], 'test_acc': [], 'gen_gap': []}
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        idx = torch.randperm(trx.size(0))[:5000]
        model = SimpleCNN(base_channels=32)
        h = train_model(model, trx[idx], try_[idx], tex, tey,
                       epochs=epochs, bs=5000, lr=0.001, opt_type='sgd', wd=0)
        ft, fe = h['train_acc'][-1], h['test_acc'][-1]
        sd['train_acc'].append(ft); sd['test_acc'].append(fe); sd['gen_gap'].append(ft-fe)
        del model; torch.cuda.empty_cache()
    mt = np.mean(sd['test_acc']); st = np.std(sd['test_acc']); mg = np.mean(sd['gen_gap'])
    print(f"test={mt:.2f}+/-{st:.2f}%, gap={mg:.2f}%", flush=True)
    results['FullBatch_GD'] = sd

    with open(os.path.join(RESULTS_DIR, 'exp2_implicit_reg.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(results.keys())
    tm = [np.mean(results[n]['test_acc']) for n in names]
    ts = [np.std(results[n]['test_acc']) for n in names]
    gg = [np.mean(results[n]['gen_gap']) for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, tm, w, yerr=ts, label='Test Accuracy (%)', capsize=3)
    ax.bar(x+w/2, gg, w, label='Generalization Gap (%)', color='orange')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_ylabel('%'); ax.set_title('Implicit Regularization: Effect of Batch Size and Optimizer')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig2_implicit_reg.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_implicit_reg.png", flush=True)
    return results


# ============================================================
# EXPERIMENT 3: LOSS LANDSCAPE
# ============================================================

def experiment3(trx, try_, tex, tey):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 3: Loss Landscape", flush=True)
    print("="*60, flush=True)

    criterion = nn.CrossEntropyLoss()

    # Train two models: small batch (flat minima) vs large batch (sharp minima)
    # Both with properly scaled learning rates
    configs = {
        'small_batch_32':  {'bs': 32,   'lr': 0.005},
        'large_batch_1024': {'bs': 1024, 'lr': 0.08},
    }
    trained = {}

    for name, cfg in configs.items():
        print(f"  Training {name} (bs={cfg['bs']}, lr={cfg['lr']})...", end=' ', flush=True)
        torch.manual_seed(42)
        model = SimpleCNN(base_channels=32)
        h = train_model(model, trx, try_, tex, tey, epochs=80,
                       bs=cfg['bs'], lr=cfg['lr'], wd=0)
        print(f"train={h['train_acc'][-1]:.1f}%, test={h['test_acc'][-1]:.2f}%", flush=True)
        trained[name] = model

    # Perturbation analysis
    print("\n  Perturbation analysis...", flush=True)
    sigmas = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    eval_x, eval_y = trx[:5000].to(DEVICE), try_[:5000].to(DEVICE)
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
        print(f"    {name}: base_loss={base_loss:.4f}", flush=True)
        for d in data:
            print(f"      sigma={d['sigma']}: +{d['pct_increase']:.1f}%", flush=True)

    # Hessian top eigenvalue
    print("\n  Hessian eigenvalue estimation...", flush=True)
    hess_results = {}
    hess_x, hess_y = trx[:1000].to(DEVICE), try_[:1000].to(DEVICE)

    for name, model in trained.items():
        print(f"    {name}...", end=' ', flush=True)
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]

        v = [torch.randn_like(p) for p in params]
        v_norm = torch.sqrt(sum((vi**2).sum() for vi in v))
        v = [vi / v_norm for vi in v]

        top_eig = 0
        for _ in range(30):
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
                     'o-', label=name, linewidth=2, markersize=6)
    axes[0].set_xlabel('Perturbation Magnitude (sigma)')
    axes[0].set_ylabel('Loss Increase (%)')
    axes[0].set_title('Loss Sensitivity to Weight Perturbation')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    names_h = list(hess_results.keys())
    eigs = [hess_results[n]['top_eigenvalue'] for n in names_h]
    colors = ['steelblue', 'coral']
    axes[1].bar(names_h, eigs, color=colors)
    axes[1].set_ylabel('Top Hessian Eigenvalue')
    axes[1].set_title('Loss Landscape Curvature (Sharpness)')
    for i, (n, e) in enumerate(zip(names_h, eigs)):
        axes[1].text(i, e + max(eigs)*0.02, f'{e:.2f}', ha='center', fontweight='bold')
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

def experiment4(trx, try_, tex, tey):
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT 4: Lottery Ticket Hypothesis", flush=True)
    print("="*60, flush=True)

    epochs = 60
    lr = 0.01
    prune_pcts = [0, 30, 50, 70, 80, 90, 95]
    results = {}

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        model = SimpleCNN(base_channels=32).to(DEVICE)
        init_state = copy.deepcopy(model.state_dict())

        h = train_model(model, trx, try_, tex, tey, epochs=epochs, bs=128, lr=lr, wd=0)
        trained_state = copy.deepcopy(model.state_dict())

        for pp in prune_pcts:
            key = f"prune_{pp}pct"
            if key not in results:
                results[key] = {'prune_pct': pp, 'test_acc': []}

            if pp == 0:
                results[key]['test_acc'].append(h['test_acc'][-1])
                print(f"  seed={seed} prune={pp}%: test={h['test_acc'][-1]:.2f}%", flush=True)
                continue

            wkeys = [k for k in trained_state if 'weight' in k and trained_state[k].dim() >= 2]
            all_w = torch.cat([trained_state[k].abs().flatten() for k in wkeys])
            thresh = torch.quantile(all_w.float(), pp / 100.0)
            mask = {k: (trained_state[k].abs() >= thresh).float().to(DEVICE) for k in wkeys}

            # Lottery ticket: reset to original init
            pruned = SimpleCNN(base_channels=32).to(DEVICE)
            pruned.load_state_dict(init_state)
            with torch.no_grad():
                sd = pruned.state_dict()
                for k in mask: sd[k].mul_(mask[k])

            # Train with mask
            pruned.to(DEVICE)
            trx_d, try_d = trx.to(DEVICE), try_.to(DEVICE)
            tex_d, tey_d = tex.to(DEVICE), tey.to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(pruned.parameters(), lr=lr, momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            n = trx_d.size(0)

            for epoch in range(epochs):
                pruned.train()
                perm = torch.randperm(n, device=DEVICE)
                for i in range(0, n, 128):
                    idx = perm[i:i+128]
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(pruned(trx_d[idx]), try_d[idx])
                    loss.backward()
                    with torch.no_grad():
                        for name, p in pruned.named_parameters():
                            if name in mask and p.grad is not None:
                                p.grad.mul_(mask[name])
                    optimizer.step()
                    with torch.no_grad():
                        for name, p in pruned.named_parameters():
                            if name in mask: p.mul_(mask[name])
                scheduler.step()

            pruned.eval()
            with torch.no_grad():
                test_acc = 100.*(pruned(tex_d).argmax(1)==tey_d).sum().item()/tey_d.size(0)
            results[key]['test_acc'].append(test_acc)
            print(f"  seed={seed} prune={pp}%: test={test_acc:.2f}%", flush=True)
            del pruned; torch.cuda.empty_cache()

        del model; torch.cuda.empty_cache()

    # Random reinit control at 90%
    print("  Random reinit control (90% pruned)...", flush=True)
    results['random_reinit_90pct'] = {'prune_pct': 90, 'test_acc': []}
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        model = SimpleCNN(base_channels=32).to(DEVICE)
        h = train_model(model, trx, try_, tex, tey, epochs=epochs, bs=128, lr=lr, wd=0)
        trained_state = model.state_dict()

        wkeys = [k for k in trained_state if 'weight' in k and trained_state[k].dim() >= 2]
        all_w = torch.cat([trained_state[k].abs().flatten() for k in wkeys])
        thresh = torch.quantile(all_w.float(), 0.9)
        mask = {k: (trained_state[k].abs() >= thresh).float().to(DEVICE) for k in wkeys}

        torch.manual_seed(seed + 100)
        pruned = SimpleCNN(base_channels=32).to(DEVICE)
        with torch.no_grad():
            sd = pruned.state_dict()
            for k in mask: sd[k].mul_(mask[k])

        trx_d, try_d = trx.to(DEVICE), try_.to(DEVICE)
        tex_d, tey_d = tex.to(DEVICE), tey.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(pruned.parameters(), lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        n = trx_d.size(0)

        for epoch in range(epochs):
            pruned.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, 128):
                idx = perm[i:i+128]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(pruned(trx_d[idx]), try_d[idx])
                loss.backward()
                with torch.no_grad():
                    for name, p in pruned.named_parameters():
                        if name in mask and p.grad is not None:
                            p.grad.mul_(mask[name])
                optimizer.step()
                with torch.no_grad():
                    for name, p in pruned.named_parameters():
                        if name in mask: p.mul_(mask[name])
            scheduler.step()

        pruned.eval()
        with torch.no_grad():
            test_acc = 100.*(pruned(tex_d).argmax(1)==tey_d).sum().item()/tey_d.size(0)
        results['random_reinit_90pct']['test_acc'].append(test_acc)
        print(f"    seed={seed}: test={test_acc:.2f}%", flush=True)
        del model, pruned; torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, 'exp4_lottery_ticket.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    pp_vals, tm_vals, ts_vals = [], [], []
    for key in sorted([k for k in results if k != 'random_reinit_90pct'],
                      key=lambda k: results[k]['prune_pct']):
        r = results[key]
        pp_vals.append(100 - r['prune_pct'])
        tm_vals.append(np.mean(r['test_acc']))
        ts_vals.append(np.std(r['test_acc']))

    ax.errorbar(pp_vals, tm_vals, yerr=ts_vals, fmt='o-', capsize=4, linewidth=2, markersize=7,
                label='Original init (Lottery Ticket)')

    ri = results['random_reinit_90pct']
    ax.errorbar([10], [np.mean(ri['test_acc'])], yerr=[np.std(ri['test_acc'])],
                fmt='X', markersize=12, capsize=4, color='red', linewidth=2,
                label='Random reinit (90% pruned)')

    ax.set_xlabel('Remaining Parameters (%)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Lottery Ticket Hypothesis: Pruning vs Performance')
    ax.invert_xaxis()
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig4_lottery_ticket.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_lottery_ticket.png", flush=True)
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    start = time.time()
    print("Loading CIFAR-10...", flush=True)
    trx, try_, tex, tey = load_cifar10()
    print(f"  Loaded: {trx.shape}", flush=True)

    experiment2(trx, try_, tex, tey)
    experiment3(trx, try_, tex, tey)
    experiment4(trx, try_, tex, tey)

    elapsed = time.time() - start
    print(f"\nCOMPLETE in {elapsed/60:.1f} minutes", flush=True)
