from sklearn.model_selection import train_test_split
from .config import config
import re

def labeled_unlabeled_test_split(df, y=None, test_size=config['test_size'], labeled_size=config['labeled_size'], random_state=config['seed']):
    strat = y if y is not None else None
    pool_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=random_state, stratify=strat
    )
    strat_pool = y[pool_idx] if strat is not None else None
    p = min(1.0, labeled_size / (1 - test_size))
    unlabeled_idx, labeled_idx = train_test_split(
        pool_idx, test_size=p, random_state=random_state, stratify=strat_pool
    )
    return labeled_idx, unlabeled_idx, test_idx

def update_indexes(data_idx, active: bool, ordered = False, config=config):
    if active:
        pass
    else:
        if ordered:
            pass
        else:
            pass # write here

def create_tokenized_df(df,token_types=config['token_types'],target_col=config['target_col'],config=config):
    def norm(s):
        return re.sub(r'\s+', ' ', str(s).strip().lower())
    def char_ngrams(s, n):
        s = norm(s)
        return [s[i:i+n] for i in range(len(s)-n+1)] if n > 0 else list(s)
    def word_tokens(s):
        return re.findall(r'\b\w+\b', norm(s))
    def word_ngrams(toks, n):
        if n <= 1:
            return toks
        return [' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)]

    token_types = token_types or config['token_types']
    out = df.copy()
    text = out[target_col].fillna('')

    for tt in token_types:
        method = tt.get('method')
        n = int(tt.get('ngram', 0))
        if method == 'char':
            n = max(1, n)
            col = f'{target_col}__char{n}'
            out[col] = text.apply(lambda s: char_ngrams(s, n))
        elif method == 'word':
            n = max(1, n)
            base = text.apply(word_tokens)
            col = f'{target_col}__word' if n == 1 else f'{target_col}__word{n}'
            out[col] = base if n == 1 else base.apply(lambda toks: word_ngrams(toks, n))

    return out