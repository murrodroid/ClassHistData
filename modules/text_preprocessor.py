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

def prepare_df_tensors(df: pd.DataFrame, column: str) -> torch.Tensor:
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

def encode_labels(df: pd.DataFrame, column: str, label_encoder=None) -> tuple:
    """
    Encodes labels from a categorical column into numeric values.
    
    If a pretrained label_encoder is provided, it is used to transform the data.
    Otherwise, a new LabelEncoder is fitted on the data.
    
    Returns:
        tuple: (tensor of encoded labels, fitted LabelEncoder)
    """
    if label_encoder is None:
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

def prepare_deathcauses_tensors(df: pd.DataFrame, column: str, token_types: list[dict]) -> tuple:
    """Prepares combined tensors for specified token types from a DataFrame column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to tokenize and prepare tensors for.
        token_types (list[dict]): A list of token types, each with 'method' and 'ngram' keys.
    
    Returns:
        tuple: A combined tensor of tokenized sequences and the combined vocabulary.
    """
    combined_vocabs = {}
    for token_type in token_types:
        # Dynamically derive name
        token_type_name = f"{token_type['method']}_{token_type['ngram']}gram" if token_type['ngram'] > 0 else f"{token_type['method']}_tokenized"
        column_name = f"{token_type_name}_tokenized"
        
        df = tokenize(df, column=column, method=token_type['method'], ngram=token_type['ngram'])
        tokenized_column = f"{column}_{token_type['method']}_{token_type['ngram']}gram" if token_type['ngram'] > 0 else f"{column}_tokenized"
        
        all_tokens = set()
        for token_list in df[tokenized_column]:
            all_tokens.update(token_list)
        
        vocab = {token: i + len(combined_vocabs) for i, token in enumerate(sorted(all_tokens))}
        combined_vocabs.update(vocab)
        
        df = df.assign(**{column_name: df[tokenized_column].apply(lambda x: [combined_vocabs[token] for token in x])})

    all_tensors = []
    for token_type in token_types:
        token_type_name = f"{token_type['method']}_{token_type['ngram']}gram" if token_type['ngram'] > 0 else f"{token_type['method']}_tokenized"
        tokenized_col = f"{token_type_name}_tokenized"
        tensor = prepare_df_tensors(df, column=tokenized_col)
        all_tensors.append(tensor)
    
    combined_tensors = torch.cat(all_tensors, dim=1)
    return combined_tensors, combined_vocabs

def train_test_split_tensors(*X, y, test_size=0.2, random_state=42):
    """
    Splits any number of input tensors (X1, X2, ...) plus a label tensor (y)
    into training and test sets, using the same random split.

    Args:
        *X (torch.Tensor): One or more input tensors to be split.
        y (torch.Tensor): Label/target tensor to be split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple of torch.Tensor: The returned tuple will contain the train-test split 
        for all provided tensors, in the order they were passed in, followed by y.

        Example (one X):
            X_train, X_test, y_train, y_test

        Example (two X's):
            X1_train, X1_test, X2_train, X2_test, y_train, y_test
    """
    # Put all tensors (X plus y) in a single list for consistent splitting
    all_tensors = list(X) + [y]

    # Basic checks
    if not all_tensors:
        raise ValueError("No tensors were provided.")

    # Ensure all tensors have the same number of samples in the first dimension
    n_samples = all_tensors[0].size(0)
    for t in all_tensors:
        if t.size(0) != n_samples:
            raise ValueError("All tensors must have the same number of samples in the first dimension.")

    # Determine how many go into train vs test
    test_samples = int(n_samples * test_size)
    train_samples = n_samples - test_samples

    # Set the random seed and shuffle indices
    torch.manual_seed(random_state)
    indices = torch.randperm(n_samples)
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]

    # Split each tensor accordingly
    split_results = []
    for t in all_tensors:
        split_results.append(t[train_indices])
        split_results.append(t[test_indices])

    # Return as a tuple so it can be unpacked
    return tuple(split_results)

def create_dataloaders(train: list, test: list, batch_size: int):
    """
    Creates PyTorch DataLoaders for training and testing datasets.

    Parameters:
    ----------
    train : list of torch.Tensor
        A list of tensors representing training data (features and labels).
    test : list of torch.Tensor
        A list of tensors representing test data (features and labels).
    batch_size : int
        Batch size for the DataLoaders.

    Returns:
    -------
    tuple (DataLoader, DataLoader)
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
    
    Raises:
    ------
    ValueError: If train and test lists do not have the same number of elements.
    TypeError: If any element in train or test is not a PyTorch tensor.
    ValueError: If tensors within train or test have mismatched first dimensions.
    """
    if len(train) != len(test):
        raise ValueError(f"Error - Mismatch in lengths: train has {len(train)} elements, test has {len(test)} elements.")

    if not all(isinstance(t, torch.Tensor) for t in train + test):
        raise TypeError("Error - All elements in train and test must be PyTorch tensors.")

    train_sizes = [t.shape[0] for t in train]
    test_sizes = [t.shape[0] for t in test]
    
    if len(set(train_sizes)) > 1:
        raise ValueError(f"Error - Mismatched sample sizes in train set: {train_sizes}")
    
    if len(set(test_sizes)) > 1:
        raise ValueError(f"Error - Mismatched sample sizes in test set: {test_sizes}")

    train_loader = DataLoader(TensorDataset(*train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(*test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader