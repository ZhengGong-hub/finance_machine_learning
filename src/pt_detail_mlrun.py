import pandas as pd
import numpy as np

# internal imports
from ml.ml_framework import MLFramework

def run_ml_for_target(target_col: str, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = '') -> str:
    """Run ML framework for a specific target column."""
    print(f"Processing target: {target_col}")
    
    # Prepare the data
    feature_cols = [col for col in train_df.columns if col.startswith('f_')]
        
    # Initialize the framework
    ml = MLFramework(train_df, test_df, target_col, feature_cols, output_dir=f'{output_dir}/ml_run_{target_col}')

    ml.prepare_data()
    
    # Train and evaluate models
    _ = ml.train_models()
    
    # Evaluate the best model
    _ = ml.evaluate_model()
    
    print(f"Completed processing target: {target_col}")
    return target_col

if __name__ == "__main__":
    # Load data once
    year = '2016'
    train_df = pd.read_parquet(f'training_data/pt_detail/{year}.parquet')
    test_df = pd.read_parquet(f'test_data/pt_detail/{year}.parquet')
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # Define target columns
    target_col = 'fwd_ret_1y_excl_div'

    run_ml_for_target(target_col, train_df, test_df, output_dir=f'output_data/ml_run_{target_col}')