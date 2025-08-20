import re
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Literal
import torch

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

def prepare_df_tensors(df: pd.DataFrame, column: str) -> torch.Tensor:
    """Prepares padded tensors from tokenized sequences.
    
    Args:
        df (pd.DataFrame): The input DataFrame with tokenized sequences.
        column (str): The name of the column containing the tokenized sequences.
    
    Returns:
        torch.Tensor: A padded tensor of tokenized sequences.
    """
    max_len = df[column].str.len().max()
    pad_id  = 0                         # ← use the reserved PAD index
    arr = np.array(
        [np.pad(seq, (0, max_len - len(seq)),
                constant_values=pad_id) for seq in df[column]],
        dtype=np.int64,
    )
    return torch.from_numpy(arr)

def encode_labels(df: pd.DataFrame, transform_column: str, label_encoder=None, fit_df: pd.DataFrame = None, fit_column: str = None) -> tuple:
    """
    Encodes labels from a categorical column into numeric values.
    
    If a pretrained label_encoder is provided, it is used to transform the data.
    Otherwise, a new LabelEncoder is fitted.
    
    If fit_df is provided, the encoder is fitted on fit_df[fit_column] (or on fit_df[transform_column] if fit_column is None),
    ensuring that the encoder sees all possible classes.
    
    The transformation (i.e. creating y_tensor) is then performed on df[transform_column].
    
    Returns:
        tuple: (tensor of encoded labels, fitted LabelEncoder)
    """
    if label_encoder is None:
        # Fit on the full data if provided
        if fit_df is not None:
            if fit_column is None:
                fit_column = transform_column
            label_encoder = LabelEncoder().fit(fit_df[fit_column])
        else:
            label_encoder = LabelEncoder().fit(df[transform_column])
    y = label_encoder.transform(df[transform_column])
    return torch.tensor(y, dtype=torch.long), label_encoder