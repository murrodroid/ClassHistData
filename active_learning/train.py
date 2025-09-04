from config import config
from import_data import import_data
from utils import update_indexes, set_seed, validate_indices
from data_store import get_features, get_df, make_initial_data_idx
from text_preprocess import tokenize
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


def train_model(data_idx, network=net, config=config, verbose=False):
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, classes = get_features()
    validate_indices(data_idx, X.shape[0])
    train_loader, _, _ = make_loaders(X, y, data_idx, batch_size=config.get('batch_size'), seed=config.get('seed'))

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

    for e in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'epoch {e+1}/{epochs}', leave=False) if verbose else train_loader
        seen, loss_sum = 0, 0.0
        for xb, yb in loop:
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            bs = yb.size(0)
            seen += bs
            loss_sum += loss.item() * bs
            if verbose:
                loop.set_postfix(avg_loss=f'{loss_sum/seen:.4f}')
    return model


def train_committee(data_idx,network=net,config=config):
    base_seed = config.get('seed', 42)
    k = config.get('committee_size')
    return [train_model(data_idx, network, {**config, 'seed': base_seed + i}) for i in range(k)]


def test_models(models, data_idx, config=config, k=None, return_both=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, _ = get_features()
    _, test_loader, _ = make_loaders(X, y, data_idx, batch_size=config.get('batch_size'), seed=config['seed'])
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
        top1_acc = (correct1 / total) if total else 0.0
        topk_acc = (correctk / total) if total else 0.0
        return top1_acc, topk_acc

    if isinstance(models, (list, tuple)):
        pairs = [acc_pair(m) for m in models]
        res = {'top1': tuple(p[0] for p in pairs), 'topk': tuple(p[1] for p in pairs), 'k': K}
    else:
        a1, ak = acc_pair(models)
        res = {'top1': (a1,), 'topk': (ak,), 'k': K}

    if return_both:
        return res
    # backwards-compat: if someone still expects a single tuple (top-K)
    return res['topk']


def train_passive(data_idx, ordered=False, verbose=False, budget=None, committee=False, config=config):
    accs, labels_hist = [], []
    df = get_df()
    R = config['rounds']
    acquired = 0
    r = 0
    target_budget = budget if budget is not None else config['rounds'] * config['query_batch_size']

    while acquired < target_budget and data_idx['unlabeled']:
        r += 1
        if verbose:
            print(f'Round {r}/{R} | labeled={len(data_idx["labeled"])} | unlabeled={len(data_idx["unlabeled"])}')

        t0 = time.perf_counter()
        if committee:
            models = train_committee(data_idx=data_idx, config=config)
            t1 = time.perf_counter()
            acc = test_models(models=models, data_idx=data_idx, config=config, return_both=True)
        else:
            model = train_model(data_idx=data_idx, config=config)
            t1 = time.perf_counter()
            acc = test_models(models=model, data_idx=data_idx, config=config, return_both=True)
        t2 = time.perf_counter()

        if verbose:
            print(f"Train {t1-t0:.2f}s | Test {t2-t1:.2f}s | "
                f"Top-1 {float(np.mean(acc['top1'])):.4f} | Top-{acc['k']} {float(np.mean(acc['topk'])):.4f}")

        accs.append(acc)
        labels_hist.append(len(data_idx['labeled']))

        k_this = min(config['query_batch_size'], target_budget - acquired, len(data_idx['unlabeled']))
        selected = select_passive_ordered(data_idx, k_this) if ordered else select_passive_random(data_idx, k=k_this, seed=config.get('seed'))

        if verbose:
            show = random.sample(selected, min(3, len(selected)))
            vals = [str(df.iloc[i][config.get('target_col','tidy_cod')]) for i in show]
            print(f'Queried {len(selected)} | Examples to label: {vals}')

        data_idx = update_indexes(data_idx, selected)

        if verbose:
            print(f'After update | labeled={len(data_idx["labeled"])} | unlabeled={len(data_idx["unlabeled"])}')

        acquired += len(selected)

    return accs, labels_hist


def train_active(data_idx,verbose=False,budget=None,ordered=False,config=config):
    accs, labels_hist = [], []
    df = get_df()
    R = config['rounds']
    acquired = 0
    r = 0
    target_budget = budget if budget is not None else config['rounds'] * config['query_batch_size']
    while acquired < target_budget and data_idx['unlabeled']:
        r += 1
        if verbose:
            print(f'Round {r}/{R} | labeled={len(data_idx["labeled"])} | unlabeled={len(data_idx["unlabeled"])}')
        t0 = time.perf_counter()
        
        committee = train_committee(data_idx=data_idx, config=config)
        t1 = time.perf_counter()
        acc = test_models(models=committee, data_idx=data_idx, config=config, return_both=True)
        t2 = time.perf_counter()

        if verbose:
            print(f"Train {t1-t0:.2f}s | Test {t2-t1:.2f}s | "
                f"Top-1 {float(np.mean(acc['top1'])):.4f} | Top-{acc['k']} {float(np.mean(acc['topk'])):.4f}")

        accs.append(acc)
        labels_hist.append(len(data_idx['labeled']))
        k_this = min(config['query_batch_size'], target_budget - acquired, len(data_idx['unlabeled']))
        s0 = time.perf_counter()
        selected = select_active(data_idx, committee, k_this, batch_size=config.get('al_batch', 4096))
        s1 = time.perf_counter()
        if verbose:
            show = random.sample(selected, min(3, len(selected)))
            vals = [str(df.iloc[i][config.get('target_col','tidy_cod')]) for i in show]
            print(f'Queried {len(selected)} | Select {s1-s0:.2f}s | Examples to label: {vals}')
        data_idx = update_indexes(data_idx, selected)
        if verbose:
            print(f'After update | labeled={len(data_idx["labeled"])} | unlabeled={len(data_idx["unlabeled"])}')
        acquired += len(selected)
    return accs, labels_hist


def train(verbose=False, save_csv=True, csv_path=None, display_figures=False, config=config):
    methods = config.get('learning_types', [])

    base_random  = make_initial_data_idx(ordered=False, cfg=config, stratify=False)
    base_ordered = make_initial_data_idx(ordered=True,  cfg=config, stratify=False)

    budget = min(
        config['rounds'] * config['query_batch_size'],
        len(base_random['unlabeled']),
        len(base_ordered['unlabeled']),
    )

    results = []
    for m in methods:
        if verbose: print(f'== {m} ==')
        if m in ('passive_random', 'active_random'):
            base = base_random
        elif m in ('passive_ordered', 'active_ordered'):
            base = base_ordered
        else:
            continue

        di = dict(labeled=list(base['labeled']), unlabeled=list(base['unlabeled']), test=list(base['test']))

        if m.startswith('passive_'):
            accs, labels_hist = train_passive(
            di,
            ordered=m.endswith('_ordered'),
            verbose=verbose,
            budget=budget,
            committee=config.get('passive_committee', False),
            config=config,
        )
        elif m.startswith('active_'):
            accs, labels_hist = train_active(di, verbose=verbose, budget=budget, config=config)
        else:
            continue

        for i, a in enumerate(accs):
            top1_tuple = tuple(float(x) for x in a['top1'])
            topk_tuple = tuple(float(x) for x in a['topk'])
            pool_size = len(base['labeled']) + len(base['unlabeled'])

            # appending each row i
            results.append({
                'method': m,
                'round': i+1,
                'labels': labels_hist[i],           # labeled **before** querying that round
                'pool_size': pool_size,             # constant per method
                'pct': 100.0 * labels_hist[i] / pool_size,
                'avg_acc_top1': float(np.mean(accs[i]['top1'])),
                'avg_acc_topk': float(np.mean(accs[i]['topk'])),
                'acc_tuple_top1': json.dumps(tuple(map(float, accs[i]['top1']))),
                'acc_tuple_topk': json.dumps(tuple(map(float, accs[i]['topk']))),
                'top_k': int(accs[i]['k']),
            })
                    
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
        if verbose: print(f'Saved: {path}')

    return df_res


### debugging

def debug_probe_window(method, pct_center=25.0, pct_window=0.6,
                       use_committee=None, config=config, fast=True,
                       verbose=True, out_dir='debug'):
    from copy import deepcopy
    cfg = deepcopy(config)
    if fast:
        cfg['epochs'] = max(1, int(cfg.get('epochs', 5) // 2))
        cfg['batch_size'] = max(16, int(cfg.get('batch_size', 128) // 2))

    base_r = make_initial_data_idx(ordered=False, cfg=cfg, stratify=False)
    base_o = make_initial_data_idx(ordered=True,  cfg=cfg, stratify=False)
    base = base_r if method.endswith('random') else base_o

    di = dict(labeled=list(base['labeled']), unlabeled=list(base['unlabeled']), test=list(base['test']))
    pool_size = len(di['labeled']) + len(di['unlabeled'])
    R = cfg['rounds']
    kq = cfg.get('query_batch_size')
    committee_flag = (use_committee
                      if use_committee is not None
                      else (method.startswith('active_') or cfg.get('passive_committee', False)))

    lo, hi = pct_center - pct_window, pct_center + pct_window
    rows, r = [], 0

    while di['unlabeled'] and r < R:
        r += 1
        labels_before = len(di['labeled'])
        pct_before = 100.0 * labels_before / pool_size

        if committee_flag:
            models = train_committee(di, config=cfg)
            metrics = test_models(models, di, config=cfg, return_both=True)
        else:
            model = train_model(di, config=cfg)
            metrics = test_models(model, di, config=cfg, return_both=True)

        top1 = tuple(map(float, metrics['top1']))
        topk = tuple(map(float, metrics['topk']))
        m1 = float(np.mean(top1)) if top1 else 0.0
        mk = float(np.mean(topk)) if topk else 0.0
        s1 = float(np.std(top1)) if len(top1) > 1 else 0.0
        sk = float(np.std(topk)) if len(topk) > 1 else 0.0

        if verbose and lo <= pct_before <= hi:
            print(f"[{method}] r={r} pct={pct_before:.2f}% labels={labels_before} "
                  f"| top1={m1:.4f}±{s1:.4f} top{metrics['k']}={mk:.4f}±{sk:.4f}")

        rows.append({
            'method': method,
            'round': r,
            'labels_before': labels_before,
            'pool_size': pool_size,
            'pct_before': pct_before,
            'top_k': int(metrics['k']),
            'top1_mean': m1, 'topk_mean': mk,
            'top1_std': s1,  'topk_std': sk,
            'top1_tuple': json.dumps(top1),
            'topk_tuple': json.dumps(topk),
        })

        if pct_before > hi:
            break

        k_this = min(kq, len(di['unlabeled']))
        if method.startswith('passive_'):
            selected = select_passive_ordered(di, k_this) if method.endswith('_ordered') \
                       else select_passive_random(di, k=k_this, seed=cfg.get('seed'))
        else:
            committee_models = models if committee_flag else [model]
            selected = select_active(di, committee_models, k_this, batch_size=cfg.get('al_batch', 4096))
        di = update_indexes(di, selected)

    df_out = pd.DataFrame(rows)
    ts = time.strftime('%Y%m%d_%H%M%S')
    pdir = Path(out_dir); pdir.mkdir(parents=True, exist_ok=True)
    fname = f"probe_{method}_pct{pct_center:.1f}_win{pct_window:.1f}_topk{cfg.get('top_k',1)}_{ts}.csv"
    path = pdir / fname
    df_out.to_csv(path, index=False)
    if verbose:
        print(f"Saved probe -> {path}")
    return df_out, path

def debug_probe_all(methods=None, pct_center=25.0, pct_window=0.6,
                    use_committee=None, config=config, fast=True,
                    verbose=True, out_dir='debug'):
    if methods is None:
        methods = ['passive_random', 'passive_ordered', 'active_random', 'active_ordered']
    dfs, paths = [], []
    for m in methods:
        dfm, pm = debug_probe_window(m, pct_center, pct_window, use_committee, config, fast, verbose, out_dir)
        dfs.append(dfm); paths.append(pm)
    df_all = pd.concat(dfs, ignore_index=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    all_path = Path(out_dir) / f"probe_all_pct{pct_center:.1f}_win{pct_window:.1f}_{ts}.csv"
    df_all.to_csv(all_path, index=False)
    if verbose:
        print(f"Saved combined -> {all_path}")
    return df_all, paths, all_path

if __name__ == "__main__":
    # probe settings
    pct_center = 25.0
    pct_window = 0.6
    methods = config.get('learning_types', ['passive_random','passive_ordered','active_random','active_ordered'])
    use_committee = None   # let the probe decide (active=True / passive_committee flag)
    fast = True            # halves epochs/batch in the probe for speed

    print(f"\n>>> targeted probe to ~{pct_center}% (±{pct_window} pp), fast={fast}\n")

    def _base_for(method):
        if method.endswith('random'):
            return make_initial_data_idx(ordered=False, cfg=config, stratify=False)
        return make_initial_data_idx(ordered=True, cfg=config, stratify=False)

    for m in methods:
        base = _base_for(m)
        L0 = len(base['labeled'])
        U0 = len(base['unlabeled'])
        pool_size = L0 + U0
        kq = int(config.get('query_batch_size', 1))
        target_labels = int(np.ceil((pct_center + pct_window) * pool_size / 100.0))
        rounds_est = max(0, int(np.ceil((target_labels - L0) / max(1, kq))))

        print(f"[plan] {m:>16} | pool={pool_size} | start_labels={L0} | batch={kq} | "
              f"est_rounds_to_window≈{rounds_est}")

    print("\nstarting probe runs...\n")

    all_paths = []
    t_all = time.perf_counter()
    for m in methods:
        t0 = time.perf_counter()
        print(f"\n--- {m} ---")
        df_probe, path_probe = debug_probe_window(
            method=m,
            pct_center=pct_center,
            pct_window=pct_window,
            use_committee=use_committee,
            config=config,
            fast=fast,
            verbose=True,
            out_dir='debug'
        )
        dt = time.perf_counter() - t0
        all_paths.append(path_probe)
        print(f"[done] {m:>16} | rows={len(df_probe)} | saved={path_probe} | elapsed={dt:.1f}s")

    dt_all = time.perf_counter() - t_all
    print("\nall probe runs finished.")
    for p in all_paths:
        print(f"  -> {p}")
    print(f"total elapsed: {dt_all:.1f}s\n")
