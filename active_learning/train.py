from config import config
from utils import update_indexes, set_seed, validate_indices, _fixed_test_loader
from data_store import get_features, get_df, make_initial_data_idx
from networks import network_hash_widedeep as net
from data_loader import make_loaders
from al_selectors import select_passive_random, select_passive_ordered, select_active

from pathlib import Path
import time, random, torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np 
import json
import random

try:
    from torch.cuda.amp import autocast as _autocast_cuda, GradScaler as _GradScaler
    def _amp_autocast(enabled): return _autocast_cuda(enabled=enabled)
except Exception:
    from torch.amp import autocast as _autocast_generic, GradScaler as _GradScaler
    def _amp_autocast(enabled): return _autocast_generic(device_type='cuda', enabled=enabled)


try:
    import wandb
    _WANDB_OK = True
except Exception:
    wandb = None
    _WANDB_OK = False

def _wb_start(cfg=config, run_name=None):
    mode = cfg.get('wandb_mode', 'disabled')
    if not _WANDB_OK or mode == 'disabled':
        return None
    if mode == 'offline':
        os.environ['WANDB_MODE'] = 'offline'
    kw = dict(project=cfg.get('wandb_project', 'active-learning'),
              config=cfg,
              name=run_name or cfg.get('wandb_run_name'))
    ent = cfg.get('wandb_entity')
    if ent: kw['entity'] = ent
    return wandb.init(**kw)

def _wb_log(run, data, step=None):
    if run is not None:
        run.log(data, step=step)


def _assert_idx_invariants(di):
    X, _, _ = get_features()
    a, b, c = set(di['labeled']), set(di['unlabeled']), set(di['test'])
    assert not (a & b or a & c or b & c)
    assert len(a) + len(b) + len(c) == X.shape[0]

def train_model(data_idx, network=net, config=config, verbose=False):
    set_seed(config.get('seed', 42))
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, classes = get_features()
    validate_indices(data_idx, X.shape[0])

    train_loader, _, _ = make_loaders(X, y, data_idx, config=config)

    model = network(
        input_dim=X.shape[1],
        num_classes=len(classes),
        hidden=config.get('hidden', 512),
        num_blocks=config.get('num_blocks', 2),
        expansion=config.get('expansion', 2),
        dropout=config.get('dropout_rate', 0.5),
        use_log1p=config.get('input_log1p', True),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.get('lr'), weight_decay=config.get('weight_decay'))
    loss_fn = nn.CrossEntropyLoss()
    epochs = config.get('epochs')

    use_amp = bool(config.get('mixed_precision', False)) and device.type == 'cuda'
    scaler = _GradScaler(enabled=use_amp)

    for e in range(epochs):
		
        model.train()
        loop = tqdm(train_loader, desc=f'epoch {e+1}/{epochs}', leave=False) if verbose else train_loader
        for xb, yb in loop:
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    return model

def train_committee(data_idx,network=net,config=config):
    base_seed = config.get('seed', 42)
    k = config.get('committee_size')
    return [train_model(data_idx, network, {**config, 'seed': base_seed + i}) for i in range(k)]


def test_models(models, data_idx, config=config, k=None, return_both=True, fixed_test_idx=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, _ = get_features()

    if fixed_test_idx is not None:
        test_loader = _fixed_test_loader(X, y, fixed_test_idx, config)
    else:
        _, test_loader, _ = make_loaders(X, y, data_idx, config=config)

    K = int(k if k is not None else config.get('top_k', 1))

    def acc_pair(m):
        m.eval()
        correct1 = correctk = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device).float(), yb.to(device)
                logits = m(xb)
                c = logits.shape[1]
                kk = max(1, min(K, c))
                top1 = logits.argmax(1)
                topk = logits.topk(kk, dim=1).indices
                correct1 += (top1 == yb).sum().item()
                correctk += topk.eq(yb.view(-1, 1)).any(dim=1).sum().item()
                total += yb.numel()
        return ((correct1 / total) if total else 0.0,
                (correctk / total) if total else 0.0)

    if isinstance(models, (list, tuple)):
        pairs = [acc_pair(m) for m in models]
        res = {'top1': tuple(p[0] for p in pairs), 'topk': tuple(p[1] for p in pairs), 'k': K}
    else:
        a1, ak = acc_pair(models)
        res = {'top1': (a1,), 'topk': (ak,), 'k': K}

    return res if return_both else res['topk']


def train_passive(data_idx, ordered=False, verbose=False, budget=None, committee=False, config=config, fixed_test_idx=None):
    accs, labels_hist, added_labels = [], [], []
    _, y, _ = get_features()
    prev_selected = np.array([], dtype=int)
    R = config['rounds']
    acquired = 0
    target_budget = budget if budget is not None else config['rounds'] * config['query_batch_size']

    while acquired < target_budget and data_idx['unlabeled']:
        added_labels.append(y[prev_selected].tolist())
        if committee:
            models = train_committee(data_idx=data_idx, config=config)
            acc = test_models(models=models, data_idx=data_idx, config=config, return_both=True, fixed_test_idx=fixed_test_idx)
        else:
            model = train_model(data_idx=data_idx, config=config)
            acc = test_models(models=model, data_idx=data_idx, config=config, return_both=True, fixed_test_idx=fixed_test_idx)
        print(f"[train_passive] Round {len(accs)}/{R} | "
        f"labeled={len(data_idx['labeled'])} | "
        f"unlabeled={len(data_idx['unlabeled'])} | "
        f"top-1={np.mean(acc['top1']):.4f} | "
        f"top-{acc['k']}={np.mean(acc['topk']):.4f}")

        accs.append(acc)
        labels_hist.append(len(data_idx['labeled']))

        k_this = min(config['query_batch_size'], target_budget - acquired, len(data_idx['unlabeled']))
        selected = select_passive_ordered(data_idx, k_this) if ordered else select_passive_random(data_idx, k=k_this, seed=config.get('seed'))
        data_idx = update_indexes(data_idx, selected)
        prev_selected = np.array(selected, dtype=int)
        acquired += len(selected)

    return accs, labels_hist, added_labels

def train_active(data_idx, verbose=False, budget=None, ordered=False, config=config, fixed_test_idx=None):
    accs, labels_hist, added_labels = [], [], []
    _, y, _ = get_features()
    prev_selected = np.array([], dtype=int)
    R = config['rounds']
    acquired = 0
    target_budget = budget if budget is not None else config['rounds'] * config['query_batch_size']

    while acquired < target_budget and data_idx['unlabeled']:
        added_labels.append(y[prev_selected].tolist())
        committee = train_committee(data_idx=data_idx, config=config)
        acc = test_models(models=committee, data_idx=data_idx, config=config, return_both=True, fixed_test_idx=fixed_test_idx)
        print(f"[train_active] Round {len(accs)}/{R} | "
        f"labeled={len(data_idx['labeled'])} | "
        f"unlabeled={len(data_idx['unlabeled'])} | "
        f"top-1={np.mean(acc['top1']):.4f} | "
        f"top-{acc['k']}={np.mean(acc['topk']):.4f}")
        accs.append(acc)
        labels_hist.append(len(data_idx['labeled']))

        k_this = min(config['query_batch_size'], target_budget - acquired, len(data_idx['unlabeled']))
        selected = select_active(data_idx, committee, k_this, batch_size=config.get('al_batch', 4096))
        data_idx = update_indexes(data_idx, selected)
        prev_selected = np.array(selected, dtype=int)
        acquired += len(selected)

    return accs, labels_hist, added_labels

def train(verbose=False, save_csv=True, csv_path=None, display_figures=False, config=config):
    methods = config.get('learning_types', [])
    base_random  = make_initial_data_idx(ordered=False, cfg=config, stratify=False)
    base_ordered = make_initial_data_idx(ordered=True,  cfg=config, stratify=False)

    run = _wb_start(config, run_name=f"AL_{time.strftime('%Y%m%d_%H%M%S')}")
    global_step = 0

    budget = min(
        config['rounds'] * config['query_batch_size'],
        len(base_random['unlabeled']),
        len(base_ordered['unlabeled'])
    )

    results = []
    for m in methods:
        print(f'starting training with method: {m}')
        base = base_random if m in ('passive_random', 'active_random') else base_ordered
        fixed_test_idx = tuple(base['test'])
        di = dict(labeled=list(base['labeled']), unlabeled=list(base['unlabeled']), test=list(base['test']))
		
        if m.startswith('passive_'):
            accs, labels_hist, added_lists = train_passive(
                di,
                ordered=m.endswith('_ordered'),
                verbose=verbose,
                budget=budget,
                committee=config.get('passive_committee', False),
                config=config,
                fixed_test_idx=fixed_test_idx,
            )
        elif m.startswith('active_'):
            accs, labels_hist, added_lists = train_active(
                di,
                verbose=verbose,
                budget=budget,
                config=config,
                fixed_test_idx=fixed_test_idx,
            )
        else:
            continue

        for i, a in enumerate(accs):
            top1_tuple = tuple(float(x) for x in a['top1'])
            topk_tuple = tuple(float(x) for x in a['topk'])
            pool_size = len(base['labeled']) + len(base['unlabeled'])
            mean1 = float(np.mean(top1_tuple)) if top1_tuple else 0.0
            meank = float(np.mean(topk_tuple)) if topk_tuple else 0.0
            std1 = float(np.std(top1_tuple)) if len(top1_tuple) > 1 else 0.0
            stdk = float(np.std(topk_tuple)) if len(topk_tuple) > 1 else 0.0
            row = {
                'method': m,
                'round': i + 1,
                'labels': labels_hist[i],
                'pool_size': pool_size,
                'pct': 100.0 * labels_hist[i] / pool_size,
                'avg_acc_top1': mean1,
                'avg_acc_topk': meank,
                'acc_tuple_top1': json.dumps(top1_tuple),
                'acc_tuple_topk': json.dumps(topk_tuple),
                'top_k': int(a['k']),
                'added_labels': json.dumps(added_lists[i]),
            }
            results.append(row)
            _wb_log(run, {
                'method': m,
                'round': row['round'],
                'labels': row['labels'],
                'pct': row['pct'],
                'acc/top1_mean': mean1,
                'acc/top1_std': std1,
                'acc/topk_mean': meank,
                'acc/topk_std': stdk,
                'top_k': row['top_k'],
                'committee_size': len(top1_tuple),
            }, step=global_step)
            global_step += 1

        if display_figures:
            xs = list(range(1, len(accs) + 1))
            ys = [float(np.mean(a['topk'])) for a in accs]
            plt.figure()
            plt.plot(xs, ys, marker='o')
            plt.title(f'{m} — top-{accs[0]["k"]} accuracy per round')
            plt.xlabel('round'); plt.ylabel('accuracy'); plt.ylim(0, 1); plt.grid(True, linestyle='--', alpha=0.35)
            plt.show()

            plt.figure()
            plt.plot(labels_hist, ys, marker='o')
            plt.title(f'{m} — top-{accs[0]["k"]} accuracy vs labels')
            plt.xlabel('labeled count'); plt.ylabel('accuracy'); plt.ylim(0, 1); plt.grid(True, linestyle='--', alpha=0.35)
            plt.show()

        if verbose:
            print(f'Final counts | labeled={len(di["labeled"])} | unlabeled={len(di["unlabeled"])} | budget={budget}')

    df_res = pd.DataFrame(results)

    if save_csv and not df_res.empty:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_dir = Path(config.get('results_dir', 'results')); out_dir.mkdir(parents=True, exist_ok=True)
        tag = (
            f'hash{config.get("hash_dim", 1<<16)}'
            f'_lr{config.get("lr")}'
            f'_bs{config.get("batch_size")}'
            f'_ep{config.get("epochs")}'
            f'_k{config.get("query_batch_size")}'
            f'_topk{config.get("top_k", 1)}'
        )
        if any(m.startswith('active_') for m in methods):
            tag += f'_comm{config.get("committee_size")}'
        if config.get('passive_committee', False) and any(m.startswith('passive_') for m in methods):
            tag += f'_passcomm{config.get("committee_size")}'
        tag += f'_budget{budget}_seed{config.get("seed")}'
        path = Path(csv_path) if csv_path else out_dir / f'al_results_{tag}_{ts}.csv'
        df_res.to_csv(path, index=False)
        if verbose:
            print(f'Saved: {path}')

    if run is not None:
        run.finish()

    return df_res

if __name__ == '__main__':
    results = train(verbose=True,save_csv=True,display_figures=False,config=config)
