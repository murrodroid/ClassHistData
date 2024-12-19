import re
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Literal

def get_word_grams(text: str, n: int, lower: bool = True, strip: bool = True) -> list:
    if lower:
        text = text.lower()
    if strip:
        text = re.sub('[^A-Za-z0-9 ]+', '', text)
    words = text.split()
    n_grams = [words[i:i + n] for i in range(len(words) - n + 1)]
    return n_grams

def get_character_grams(word: str, n: int) -> list:
    word = f'<{word}>'
    return [word[i:i + n] for i in range(len(word) - n + 1)]

def tokenize(df: pd.DataFrame, column: str, method: Literal['char', 'word'], ngram: int = 0) -> pd.DataFrame:
    if method == 'char':
        if ngram > 0:
            df[f'{column}_char_{ngram}gram'] = df[column].apply(lambda x: [ngram for ngram in get_character_grams(x, n=ngram)])
        else:
            unique_chars = set(''.join(df[column]))
            char_tokenize = {char: i for i, char in enumerate(unique_chars)}
            df[f'{column}_tokenized'] = df[column].apply(lambda x: [char_tokenize[char] for char in x])
    elif method == 'word':
        all_words = set()
        for text in df[column]:
            all_words.update(text.split())
        word_tokenize = {word: i for i, word in enumerate(sorted(all_words))}
        if ngram > 0:
            df[f'{column}_word_{ngram}gram'] = df[column].apply(lambda x: get_word_grams(x, n=ngram))
        else:
            df[f'{column}_tokenized'] = df[column].apply(lambda x: [word_tokenize[word] for word in x.split()])
    return df

def prepare_tensors(df: pd.DataFrame, column: str) -> torch.Tensor:
    max_len = max(df[column].apply(len))
    padded_sequences = np.array([
        np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=0)
        for seq in df[column]
    ])
    return torch.tensor(padded_sequences, dtype=torch.long)

def encode_labels(df: pd.DataFrame, column: str) -> torch.Tensor:
    label_encoder = LabelEncoder().fit(df[column])
    y = label_encoder.transform(df[column])
    return torch.tensor(y, dtype=torch.long), label_encoder

def print_model_parameters(df: pd.DataFrame, char_vocab: dict, word_vocab: dict, ngram_vocab: dict, label_encoder: LabelEncoder) -> None:
    print("Character Vocab Size:", len(char_vocab))
    print("N-Gram Vocab Size:", len(ngram_vocab))
    print("Word Vocab Size:", len(word_vocab))
    print("Number of Output Classes:", len(label_encoder.classes_))

def prepare_combined_tensors(df: pd.DataFrame, column: str) -> tuple:
    token_types = [
        {'method': 'char', 'ngram': 1, 'name': 'char_unigram'},
        {'method': 'char', 'ngram': 2, 'name': 'char_bigram'},
        {'method': 'char', 'ngram': 3, 'name': 'char_trigram'},
        {'method': 'word', 'ngram': 0, 'name': 'word_gram'}
    ]
    
    combined_vocabs = {}
    for token_type in token_types:
        column_name = f"{token_type['name']}_tokenized"
        df = tokenize(df, column=column, method=token_type['method'], ngram=token_type['ngram'])
        tokenized_column = f"{column}_{token_type['method']}_{token_type['ngram']}gram" if token_type['ngram'] > 0 else f'{column}_tokenized'
        
        all_tokens = set()
        for token_list in df[tokenized_column]:
            all_tokens.update(token_list)
        
        vocab = {token: i + len(combined_vocabs) for i, token in enumerate(sorted(all_tokens))}
        combined_vocabs.update(vocab)
        
        df[column_name] = df[tokenized_column].apply(lambda x: [combined_vocabs[token] for token in x])
    
    all_tensors = []
    for token_type in token_types:
        tokenized_col = f"{token_type['name']}_tokenized"
        tensor = prepare_tensors(df, column=tokenized_col)
        all_tensors.append(tensor)
    
    combined_tensors = torch.cat(all_tensors, dim=1)
    return combined_tensors, combined_vocabs