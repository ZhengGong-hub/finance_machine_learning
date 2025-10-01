import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import os

# internal imports
from logger import setup_logger
from utils.remove_row_with_missing_values import remove_row_with_missing_values
from utils.helper import get_last_three_years_yq
# setup logger
logger = setup_logger(__name__)

if __name__ == "__main__":
    # Create directories
    training_data_dir = 'training_data/pt_summary'
    test_data_dir = 'test_data/pt_summary'
    os.makedirs(training_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Load data
    df = pd.read_parquet('input_data/all_data_with_pt_summary.parquet')# .query("permno == 14593")
    logger.info(f"columns: {df.columns}")
    logger.info(f"Original data shape: {df.shape}")
    df['year'] = df['trading_day_et'].dt.year
    df['quarter'] = df['trading_day_et'].dt.quarter

    # drop row if the fwd_ret is missing
    df = df.dropna(subset=['fwd_ret_1y_excl_div', 'medptg'], how='any')
    # the forward return should be bigger than -1 (namely lost everything)
    df = df[df['fwd_ret_1y_excl_div'] > -1]
    # drop row if marketcap is prc < 1
    df = df[df['prc'] >= 1]
    # drop row if marketcap <= 0
    df = df[df['marketcap'] > 0]
    # drop row if btm is nan, i.e. no book equity
    df = df.dropna(subset=['f_btm', 'f_sp', 'f_ep'], how='any')
    # marketcap per year we drop the smallest 20%
    df = df.groupby('trading_day_et').apply(lambda x: x.sort_values(by='marketcap', ascending=True).tail(int(len(x) * 0.8))).reset_index(drop=True)
    # replace inf and -inf with nan
    f_cols = [col for col in df.columns if col.startswith('f_')]
    df[f_cols] = df[f_cols].replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"after price > 1 and marketcap > 0 and required fundamentals are not missing. data shape: {df.shape}")
    logger.info("peek at the data")
    logger.info(df.head())

    # for ravenpack columns, impute with 0
    # select the columns that start with f_rp_
    rp_cols = [col for col in df.columns if col.startswith('f_rp_')]
    df[rp_cols] = df[rp_cols].fillna(0)

    # drop rows with missing values
    df_processed = remove_row_with_missing_values(df, threshold=10)
    logger.info(f"Processed data shape: {df_processed.shape}")

    # impute missing values by cross-sectional median
    # select the columns that start with f_
    f_cols = [col for col in df_processed.columns if col.startswith('f_')]
    df_processed[f_cols] = df_processed.groupby('trading_day_et')[f_cols].apply(lambda x: x.fillna(x.median())).reset_index(drop=True)

    # construct y variable
    df_processed['pt_return'] = df_processed['medptg']/df_processed['prc'] - 1
    df_processed['error'] = np.log(1 + df_processed['pt_return']) - np.log(1 + df_processed['fwd_ret_1y_excl_div'])
    
    # separate into train and test
    # use last three years to train and test on the coming quarter
    year_start = df_processed['year'].min()
    year_end = df_processed['year'].max()

    # For each year and quarter, create rolling training and test sets.
    # The training set consists of the previous three years (by quarter) up to but not including the current quarter.
    # The test set is the current year and quarter.
    for year in range(year_start + 3, year_end + 1):
        for quarter in range(1, 5):
            logger.info(f"Processing rolling window for year={year}, quarter={quarter}")
            last_three_years_yq = get_last_three_years_yq([year, quarter])
            logger.info(f"Training set will include data from the following year-quarters: {last_three_years_yq}")

            # Select rows where (year, quarter) is in last_three_years_yq for training
            train_set = df_processed[
                df_processed[['year', 'quarter']].apply(tuple, axis=1).isin([tuple(yq) for yq in last_three_years_yq])
            ]
            logger.info(f"Training set shape for {year}Q{quarter}: {train_set.shape}")
            train_set.to_parquet(f'{training_data_dir}/{year}_{quarter}.parquet', index=False)

            # Select rows for the current year and quarter for testing
            test_set = df_processed[
                (df_processed['year'] == year) & (df_processed['quarter'] == quarter)
            ]
            logger.info(f"Test set shape for {year}Q{quarter}: {test_set.shape}")
            test_set.to_parquet(f'{test_data_dir}/{year}_{quarter}.parquet', index=False)






    


    
