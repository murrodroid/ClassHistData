import re
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Literal
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn

def get_word_grams(text: str, n: int, lower: bool = True, strip: bool = True) -> list:
    """Generates n-grams from words in the given text string.
    
    Args:
        text (str): The input text string.
        n (int): The size of the n-grams.
        lower (bool): Whether to convert the text to lowercase.
        strip (bool): Whether to remove non-alphanumeric characters from the text.
    
    Returns:
        list: A list of n-grams.
    """
    if lower:
        text = text.lower()
    if strip:
        text = re.sub('[^A-Za-z0-9 ]+', '', text)
    words = text.split()
    n_grams = [words[i:i + n] for i in range(len(words) - n + 1)]
    return n_grams

def get_character_grams(word: str, n: int) -> list:
    """Generates n-grams of characters for a given word.
    
    Args:
        word (str): The input word.
        n (int): The size of the character n-grams.
    
    Returns:
        list: A list of character n-grams.
    """
    word = f'<{word}>'
    return [word[i:i + n] for i in range(len(word) - n + 1)]

def tokenize(df: pd.DataFrame, column: str, method: Literal['char', 'word'], ngram: int = 0) -> pd.DataFrame:
    """Tokenizes a column in a DataFrame using character or word tokens and optional n-grams.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to tokenize.
        method (Literal): The type of tokenization ('char' or 'word').
        ngram (int): The size of the n-gram. Defaults to 0 for no n-grams.
    
    Returns:
        pd.DataFrame: The DataFrame with an additional tokenized column.
    """
    if method == 'char':
        if ngram > 0:
            df = df.assign(**{f'{column}_char_{ngram}gram': df[column].apply(lambda x: [ngram for ngram in get_character_grams(x, n=ngram)])})
        else:
            unique_chars = set(''.join(df[column]))
            char_tokenize = {char: i for i, char in enumerate(unique_chars)}
            df = df.assign(**{f'{column}_tokenized': df[column].apply(lambda x: [char_tokenize[char] for char in x])})
    elif method == 'word':
        all_words = set()
        for text in df[column]:
            all_words.update(text.split())
        word_tokenize = {word: i for i, word in enumerate(sorted(all_words))}
        if ngram > 0:
            df = df.assign(**{f'{column}_word_{ngram}gram': df[column].apply(lambda x: get_word_grams(x, n=ngram))})
        else:
            df = df.assign(**{f'{column}_tokenized': df[column].apply(lambda x: [word_tokenize[word] for word in x.split()])})
    return df

def prepare_tensors(df: pd.DataFrame, column: str) -> torch.Tensor:
    """Prepares padded tensors from tokenized sequences.
    
    Args:
        df (pd.DataFrame): The input DataFrame with tokenized sequences.
        column (str): The name of the column containing the tokenized sequences.
    
    Returns:
        torch.Tensor: A padded tensor of tokenized sequences.
    """
    max_len = max(df[column].apply(len))
    padded_sequences = np.array([
        np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=0)
        for seq in df[column]
    ])
    return torch.tensor(padded_sequences, dtype=torch.long)

def encode_labels(df: pd.DataFrame, column: str) -> torch.Tensor:
    """Encodes labels from a categorical column into numeric values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column containing categorical labels.
    
    Returns:
        tuple: A tensor of encoded labels and the fitted LabelEncoder.
    """
    label_encoder = LabelEncoder().fit(df[column])
    y = label_encoder.transform(df[column])
    return torch.tensor(y, dtype=torch.long), label_encoder

def print_model_parameters(df: pd.DataFrame, char_vocab: dict, word_vocab: dict, ngram_vocab: dict, label_encoder: LabelEncoder) -> None:
    """Prints the model's vocabulary sizes and the number of output classes.
    
    Args:
        df (pd.DataFrame): The DataFrame used for tokenization.
        char_vocab (dict): The character-level vocabulary.
        word_vocab (dict): The word-level vocabulary.
        ngram_vocab (dict): The n-gram-level vocabulary.
        label_encoder (LabelEncoder): The LabelEncoder used for encoding labels.
    """
    print("Character Vocab Size:", len(char_vocab))
    print("N-Gram Vocab Size:", len(ngram_vocab))
    print("Word Vocab Size:", len(word_vocab))
    print("Number of Output Classes:", len(label_encoder.classes_))

def prepare_combined_tensors(df: pd.DataFrame, column: str) -> tuple:
    """Prepares combined tensors for unigrams, bigrams, trigrams, and word grams from a DataFrame column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to tokenize and prepare tensors for.
    
    Returns:
        tuple: A combined tensor of tokenized sequences and the combined vocabulary.
    """
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
        
        df = df.assign(**{column_name: df[tokenized_column].apply(lambda x: [combined_vocabs[token] for token in x])})
    
    all_tensors = []
    for token_type in token_types:
        tokenized_col = f"{token_type['name']}_tokenized"
        tensor = prepare_tensors(df, column=tokenized_col)
        all_tensors.append(tensor)
    
    combined_tensors = torch.cat(all_tensors, dim=1)
    return combined_tensors, combined_vocabs

def train_test_split_tensors(X, y, test_size=0.2, random_state=42):
    """
    Splits the input and label tensors into training and test sets.
    
    Args:
        X (torch.Tensor): Combined input tensor.
        y (torch.Tensor): Encoded target labels.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    total_samples = X.size(0)
    test_samples = int(total_samples * test_size)
    train_samples = total_samples - test_samples
    
    torch.manual_seed(random_state)
    indices = torch.randperm(total_samples)
    
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test