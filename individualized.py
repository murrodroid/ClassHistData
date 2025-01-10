# packages
import torch
import torch.nn as nn
# modules
from modules.import_data import import_data_individualized
from modules.text_preprocessor import *
from sklearn.preprocessing import StandardScaler

df, _ = import_data_individualized()

val_sample = df.sample(n=round(df.shape[0]*0.002),random_state=333)
df = df.drop(val_sample.index)
train_df = df[df.icd10h.notna()]

token_types = [
        #{'method': 'char', 'ngram': 1},
        {'method': 'char', 'ngram': 2},
        {'method': 'char', 'ngram': 3},
        {'method': 'word', 'ngram': 0}
    ]

# should we train on deathcause_mono or deathcauses?
X_cause, vocab = prepare_combined_tensors(df=train_df,column='deathcause_mono',token_types=token_types)
X_age  = StandardScaler().fit_transform(train_df['age'].to_numpy().reshape(-1,1))



