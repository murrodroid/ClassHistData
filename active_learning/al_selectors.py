import numpy as np, torch, random
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling
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

def select_active(data_idx, committee_models, k, batch_size=2048):
    X, _, classes = get_features()
    ul = data_idx['unlabeled']
    if not ul or k <= 0: return []
    learners = [ActiveLearner(estimator=TorchEstimator(m, len(classes), batch_size=batch_size)) for m in committee_models]
    committee = Committee(learner_list=learners, query_strategy=vote_entropy_sampling)
    pool = X[ul]
    ask_rel, _ = committee.query(pool, n_instances=min(k, len(ul)))
    return [ul[i] for i in (ask_rel.tolist() if hasattr(ask_rel, 'tolist') else list(ask_rel))]
