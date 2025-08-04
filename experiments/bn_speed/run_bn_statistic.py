# Usage:  python -m experiments.bn_speed.run_bn_statistic --epochs 14 --runs 20

import argparse, pathlib, csv
import torch, pandas as pd, matplotlib.pyplot as plt

from src.models import BaseCNN, CNN_BN
from src.dataloaders import get_dataloaders
from src.train_utils import set_seed, train_one_epoch, test_epoch


def train_and_log(model_cls, tag, run_id, epochs, lr, batch, device, out_dir, save_model):
    """Train one run of one model; log epoch‑wise accuracy."""
    set_seed(1 + run_id)

    tr_loader, te_loader = get_dataloaders(batch, 1000)
    model = model_cls().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    csv_path = out_dir / f'stats_{tag}_run{run_id}.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['epoch', 'train_acc', 'test_acc'])
        for ep in range(1, epochs + 1):
            tr_loss, tr_acc, tr_time = train_one_epoch(model, tr_loader, optim, device)
            te_loss, te_acc    = test_epoch(model,  te_loader, device)
            # This uses the test procedure on the training set.
            # As is, training accuracy is lower than test accuracy.
            # If you run this instead this is not the case.
            # This confirms, that the cause is improvement during an epoch of the model, dropout and the different batch norm behaviour
            # tr_loss, tr_acc    = test_epoch(model,  tr_loader, device)  
            w.writerow([ep, tr_acc, te_acc])
            print(f'{tag}  epoch {ep}:  train {tr_acc*100:.2f}%  test {te_acc*100:.2f}%')
    if save_model:
        torch.save(model.state_dict(), out_dir / f'cnn_{tag}.pt')     
    return csv_path


def aggregate_runs(csv_paths):
    """Return DataFrame with mean ± std over runs."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    cat = pd.concat(dfs).groupby('epoch').agg(['mean', 'std'])
    # flatten MultiIndex columns -> ('train_acc','mean') -> 'train_acc_mean'
    cat.columns = ['_'.join(c) for c in cat.columns]
    return cat.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--lr',     type=float, default=0.01)
    ap.add_argument('--batch',  type=int, default=64)
    ap.add_argument('--runs',   type=int, default=50)
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    paths_base, paths_bn = [], []

    for run in range(args.runs):
        print(f'Run {run} of {args.runs}')
        save_model = (run == 0)
        paths_base.append(train_and_log(BaseCNN, 'base', run,
                                        args.epochs, args.lr, args.batch, dev, root, save_model))
        paths_bn.append(train_and_log(CNN_BN,  'bn',   run,
                                        args.epochs, args.lr, args.batch, dev, root, save_model))

    # --- aggregate & plot ---------------------------------------------------
    base = aggregate_runs(paths_base)
    bn   = aggregate_runs(paths_bn)

    plt.figure(figsize=(6,3))
    plt.plot(base['epoch'], base['test_acc_mean'], label='BaseCNN')
    plt.fill_between(base['epoch'],
                     base['test_acc_mean']-base['test_acc_std'],
                     base['test_acc_mean']+base['test_acc_std'],
                     alpha=0.2)

    plt.plot(bn['epoch'],   bn['test_acc_mean'], label='CNN_BN')
    plt.fill_between(bn['epoch'],
                     bn['test_acc_mean']-bn['test_acc_std'],
                     bn['test_acc_mean']+bn['test_acc_std'],
                     alpha=0.2)

    plt.ylabel('test accuracy'); plt.xlabel('epoch')
    plt.title(f'MNIST – mean $\pm 1 \sigma$ over {args.runs} runs')
    plt.legend()
    out_png = root / f'accuracy_compare_{args.runs}runs.png'      # CHG
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print('Saved plot →', out_png)


if __name__ == '__main__':
    main()
