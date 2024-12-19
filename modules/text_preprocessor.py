import re
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Literal

def get_word_grams(text: str, n: int, lower: bool = True, strip: bool = True) -> list:
    """Generates n-grams from words in the given text string.
    
    Args:
        text (str): The input text string.
        n (int): The size of the n-grams.
        lower (bool, optional): Whether to convert text to lowercase. Defaults to True.
        strip (bool, optional): Whether to remove non-alphanumeric characters. Defaults to True.

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
    """Generates character n-grams from a single word.
    
    Args:
        word (str): The input word.
        n (int): The size of the n-grams.

    Returns:
        list: A list of character n-grams.
    """
    word = f'<{word}>'
    return [word[i:i + n] for i in range(len(word) - n + 1)]

def tokenize(df: pd.DataFrame, column: str, method: Literal['char', 'word'], ngram: int = 0) -> pd.DataFrame:
    """Tokenizes a column of a DataFrame based on characters, words, or n-grams.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to tokenize.
        method (str): The type of tokenization ('char' or 'word').
        ngram (int, optional): The size of the n-gram. Use 0 for no n-gram. Defaults to 0.
    
    Returns:
        pd.DataFrame: The DataFrame with the tokenized column.
    """
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
    """Prepares tensors for tokenized sequences, padding them to the same length.
    
    Args:
        df (pd.DataFrame): The DataFrame with tokenized sequences.
        column (str): The column containing the tokenized sequences.
    
    Returns:
        torch.Tensor: A padded tensor with tokenized sequences.
    """
    max_len = max(df[column].apply(len))
    padded_sequences = np.array([
        np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=0)
        for seq in df[column]
    ])
    
    return torch.tensor(padded_sequences, dtype=torch.long)

def encode_labels(df: pd.DataFrame, column: str) -> torch.Tensor:
    """Encodes categorical labels into numerical labels using LabelEncoder.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the labels to encode.
        column (str): The name of the column to encode.
    
    Returns:
        torch.Tensor: A tensor with encoded labels.
    """
    label_encoder = LabelEncoder().fit(df[column])
    y = label_encoder.transform(df[column])
    
    return torch.tensor(y, dtype=torch.long), label_encoder

def print_model_parameters(df: pd.DataFrame, char_vocab: dict, word_vocab: dict, ngram_vocab: dict, label_encoder: LabelEncoder) -> None:
    """Prints model parameter statistics like vocabulary sizes and number of classes.
    
    Args:
        df (pd.DataFrame): The DataFrame with data.
        char_vocab (dict): Character-level vocabulary.
        word_vocab (dict): Word-level vocabulary.
        ngram_vocab (dict): N-gram vocabulary.
        label_encoder (LabelEncoder): Encoder for target labels.
    """
    print("Character Vocab Size:", len(char_vocab))
    print("N-Gram Vocab Size:", len(ngram_vocab))
    print("Word Vocab Size:", len(word_vocab))
    print("Number of Output Classes:", len(label_encoder.classes_))
