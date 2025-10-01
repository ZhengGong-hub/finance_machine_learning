import pandas as pd
from typing import Tuple, List, Set

# imports
from logger import setup_logger

# logger setup
logger = setup_logger(__name__)

def remove_row_with_missing_values(
    df: pd.DataFrame, 
    threshold: float = 50.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove rows with missing values exceeding the threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean
    threshold : float, default=50.0
        Maximum percentage of missing values allowed in a row
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Clean DataFrame with high-missing-value rows removed
        - List of rows that were dropped
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    if len(df.columns) == 0:
        raise ValueError("Input DataFrame has no columns")
        
    if not 0 <= threshold <= 100:
        raise ValueError(f"Threshold must be between 0 and 100, got {threshold}")
    
    # working copy
    df_clean = df.copy()
    
    # calculate how many columns are missing
    # only look at the columbns that starts with 'f_'
    f_columns = [col for col in df_clean.columns if col.startswith('f_')]
    df_clean['missing_pct'] = df_clean[f_columns].isnull().mean(axis=1) * 100
    df_clean = df_clean.query(f'missing_pct <= {threshold}')

    logger.info(f"Dropped {df.shape[0] - df_clean.shape[0]} rows with missing values")
    logger.info(f"Processed data shape: {df_clean.shape}")
    
    return df_clean