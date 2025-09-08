import numpy as np, torch, random
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling, consensus_entropy_sampling
from scipy.sparse import issparse
from data_store import get_features

class TorchEstimator:
    def __init__(self, model, num_classes, device=None, batch_size=2048):
        self.model = model.eval().to(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.classes_ = np.arange(num_classes)

    def predict_proba(self, X):
        if issparse(X): X = X.tocsr()
        n = X.shape[0]
        out = []
        for i in range(0, n, self.batch_size):
            Xi = X[i:i+self.batch_size]
            if hasattr(Xi, 'toarray'): Xi = Xi.toarray()
            xb = torch.from_numpy(np.asarray(Xi, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                p = self.model(xb).softmax(1).cpu().numpy()
            out.append(p)
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

def select_passive_random(data_idx, k, seed):
    k = min(k, len(data_idx['unlabeled']))
    random.seed(seed)
    return random.sample(data_idx['unlabeled'], k)

def select_passive_ordered(data_idx, k):
    ul = np.asarray(data_idx['unlabeled'], dtype=int)
    if ul.size == 0 or k <= 0:
        return []
    return np.sort(ul)[:min(k, ul.size)].tolist()

def _diverse_maxmin(C, k):
    from scipy.sparse import issparse
    if issparse(C):
        C = C.tocsr()
        norms = np.sqrt(C.multiply(C).sum(1)).A1
        norms[norms == 0] = 1.0
        Cn = C.multiply(1.0 / norms[:, None])
        n = Cn.shape[0]
        sel = [0]
        sim = (Cn @ Cn[0].T).A1
        for _ in range(1, min(k, n)):
            dist = 1.0 - sim
            j = int(dist.argmax())
            sel.append(j)
            sim = np.maximum(sim, (Cn @ Cn[j].T).A1)
        return sel
    C = np.asarray(C)
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Cn = C / norms
    n = Cn.shape[0]
    sel = [0]
    sim = Cn @ Cn[0]
    for _ in range(1, min(k, n)):
        dist = 1.0 - sim
        j = int(dist.argmax())
        sel.append(j)
        sim = np.maximum(sim, Cn @ Cn[j])
    return sel

def select_active(data_idx, committee_models, k, batch_size=2048, diversity_factor=10):
    X, _, classes = get_features()
    ul = np.asarray(data_idx['unlabeled'], dtype=int)
    if ul.size == 0 or k <= 0:
        return []
    learners = [ActiveLearner(estimator=TorchEstimator(m, len(classes), batch_size=batch_size)) for m in committee_models]
    committee = Committee(learner_list=learners, query_strategy=vote_entropy_sampling)
    pool = X[ul]
    m = int(min(ul.size, max(k, k * diversity_factor)))
    rel_idx, _ = committee.query(pool, n_instances=m)
    rel_idx = rel_idx.tolist() if hasattr(rel_idx, 'tolist') else list(rel_idx)
    if len(rel_idx) <= k:
        return ul[rel_idx].tolist()
    C = pool[rel_idx]
    pick_rel = _diverse_maxmin(C, k)
    return [int(ul[rel_idx[i]]) for i in pick_rel]