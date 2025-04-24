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

def prepare_deathcauses_tensors(
    df: pd.DataFrame,
    column: str,
    token_types: list[dict],
    pretrained_vocab: dict | None = None,
):
    """
    Build one big vocabulary shared by all token types, reserve id 0 for PAD,
    map every sequence to integer ids, pad to max-length, and concatenate the
    tensors along the time dimension.

    Returns
    -------
    combined_tensor : torch.Tensor   # (N, L_total)
    vocab           : dict           # token -> id (0 = PAD)
    """
    # ───── 1. build / load vocab ──────────────────────────────────────────
    if pretrained_vocab is None:
        vocab = {"<PAD>": 0}                   # 0 = padding
        next_id = 1

        for t in token_types:
            # tokenize the column if not already present
            df = tokenize(df, column, t["method"], t["ngram"])
            tok_col = (
                f"{column}_{t['method']}_{t['ngram']}gram"
                if t["ngram"] > 0
                else f"{column}_tokenized"
            )

            for token in df[tok_col]:
                for tok in token:              # token is a list
                    if tok not in vocab:
                        vocab[tok] = next_id
                        next_id += 1
    else:
        vocab = pretrained_vocab

    # ───── 2. map tokens → ids and pad each sequence ─────────────────────
    tensors = []
    for t in token_types:
        df = tokenize(df, column, t["method"], t["ngram"])
        src_col = (
            f"{column}_{t['method']}_{t['ngram']}gram"
            if t["ngram"] > 0
            else f"{column}_tokenized"
        )
        dst_col = f"{t['method']}_{t['ngram']}gram_tokenized"

        df[dst_col] = df[src_col].apply(
            lambda seq: [vocab[tok] for tok in seq if tok in vocab]
        )

        # pad to max-length of this token-type
        max_len = df[dst_col].str.len().max()
        pad_id = 0                             # we reserved 0 for PAD
        padded = np.array(
            [
                np.pad(s, (0, max_len - len(s)), constant_values=pad_id)
                for s in df[dst_col]
            ],
            dtype=np.int64,
        )
        tensors.append(torch.from_numpy(padded))

    # ───── 3. concat along time dimension ────────────────────────────────
    combined = torch.cat(tensors, dim=1)       # (batch, L1+L2+…)
    return combined, vocab

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

def create_dataloaders(train: list, test: list = None, batch_size: int = 32):
    """
    Creates PyTorch DataLoaders for training and testing (or validation) datasets.

    Parameters:
    ----------
    train : list of torch.Tensor
        A list of tensors representing training data (features and labels).
    test : list of torch.Tensor, optional
        A list of tensors representing test/validation data (features and labels).
        If not provided, only a single DataLoader is created for the training list.
    batch_size : int
        Batch size for the DataLoaders.

    Returns:
    -------
    tuple:
        If test is provided:
            (train_loader, test_loader)
        Otherwise:
            (val_loader) where val_loader is a DataLoader created from train list.
    
    Raises:
    ------
    ValueError: If train and test lists do not have the same number of elements.
    TypeError: If any element in train or test is not a PyTorch tensor.
    ValueError: If tensors within train or test have mismatched first dimensions.
    """
    # Check that train elements are tensors
    if not all(isinstance(t, torch.Tensor) for t in train):
        raise TypeError("Error - All elements in train must be PyTorch tensors.")
        
    train_sizes = [t.shape[0] for t in train]
    if len(set(train_sizes)) > 1:
        raise ValueError(f"Error - Mismatched sample sizes in train set: {train_sizes}")
    
    if test is not None:
        if not all(isinstance(t, torch.Tensor) for t in test):
            raise TypeError("Error - All elements in test must be PyTorch tensors.")
        test_sizes = [t.shape[0] for t in test]
        if len(set(test_sizes)) > 1:
            raise ValueError(f"Error - Mismatched sample sizes in test set: {test_sizes}")
        if len(train) != len(test):
            raise ValueError(f"Error - Mismatch in lengths: train has {len(train)} elements, test has {len(test)} elements.")

        train_loader = DataLoader(TensorDataset(*train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(*test), batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        # Create a single DataLoader (e.g. for validation)
        val_loader = DataLoader(TensorDataset(*train), batch_size=batch_size, shuffle=False)
        return val_loader