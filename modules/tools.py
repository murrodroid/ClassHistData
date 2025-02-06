import torch
import pandas as pd

def remove_text_after_comma(s):
    paren_level = 0
    for i, c in enumerate(s):
        if c == '(':
            paren_level += 1
        elif c == ')':
            paren_level -= 1
        elif c == ',' and paren_level == 0:
            return s[:i]
    return s

def return_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def undersampling(df, target_col='icd10h', scale=0.1, lower_bound=50, random_state=333, verbose=False):
    """
    Undersamples the DataFrame so that each class in the target column is reduced by a fixed fraction,
    but not below a specified minimum number of samples (unless the class originally has fewer samples).
    
    For each class:
        - If the original count n is less than lower_bound, all samples are kept.
        - Otherwise, the final count is:
              final_count = min(n, max(lower_bound, int(n * scale)))
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    target_col : str, default 'icd10h'
        The name of the column used for class labels.
    scale : float, default 0.1
        The fraction of samples to keep for each class (before applying the lower bound).
        (Should be between 0 and 1 for undersampling.)
    lower_bound : int, default 50
        The minimum number of samples to keep for any class (if available).
    random_state : int, default 333
        Random seed for reproducibility.
    verbose : bool, default False
        If True, prints the original and new class distributions.
        
    Returns
    -------
    balanced_df : pandas.DataFrame
        The undersampled DataFrame.
        
    Raises
    ------
    ValueError
        If the target column is not found or if parameters are out of expected ranges.
    """
    # Check that the target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Check parameter validity
    if not (0 < scale <= 1):
        raise ValueError("Parameter 'scale' should be in the range (0, 1].")
    if lower_bound < 1:
        raise ValueError("Parameter 'lower_bound' must be at least 1.")
    
    if verbose:
        print("Original class distribution:")
        print(df[target_col].value_counts())
    
    # Container for the undersampled subsets
    sampled_dfs = []
    
    # Get the counts per class
    class_counts = df[target_col].value_counts()
    
    for label, n in class_counts.items():
        class_subset = df[df[target_col] == label]
        
        # If the class has fewer samples than lower_bound, keep them all.
        if n <= lower_bound:
            final_count = n
        else:
            # Compute the scaled count and then ensure at least lower_bound samples.
            scaled_count = int(n * scale)
            final_count = max(lower_bound, scaled_count)
            # Ensure we don't sample more than available.
            final_count = min(n, final_count)
        
        # Sample final_count rows (if final_count < n, perform sampling)
        if final_count < n:
            class_subset = class_subset.sample(n=final_count, random_state=random_state)
        
        sampled_dfs.append(class_subset)
    
    # Combine the sampled subsets and shuffle the final DataFrame
    balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    if verbose:
        print("\nNew class distribution:")
        print(balanced_df[target_col].value_counts())
        print(f'Ratio: {balanced_df.shape[0]/df.shape[0]}')
        print(f'Absolute: {balanced_df.shape[0]} / {df.shape[0]}')
    
    return balanced_df