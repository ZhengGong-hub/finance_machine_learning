#    transcriptid   keydevid  companyid  fiscalyear  fiscalquarter  ...     SP  SalesAcc  SolvencyRatio  StdErr180D WCTurn
# 0        261760  143123704      27685      2011.0            4.0  ...  0.884     1.051          0.598      31.881  7.433
# 1        309189  170941554      27685      2012.0            1.0  ...  0.813    -0.416          0.619      12.547  7.723
# 2        347564  170941641      27685      2012.0            2.0  ...  0.736    -0.192          0.653      17.110  7.616
# 3        373214  222711696     251704      2012.0            1.0  ...  4.305     7.450          0.119     -32.358  6.892
# 4        387960  170941703      27685      2012.0            3.0  ...  0.632    -1.052          0.669      52.331  7.487

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
from typing import List, Dict, Union, Optional, Any
import logging
import os
from datetime import datetime

# internal imports
from logger import setup_logger

# import yaml
import yaml

# load nn.yaml
with open('src/ml/nn.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

# setup logger
logger = setup_logger(__name__)

class MLFramework:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, feature_cols: List[str], output_dir: str = 'output_data/ml_run'):
        """Initialize the ML framework with a pandas DataFrame."""
        self.train_df = train_df
        self.test_df = test_df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.target_col = target_col
        self.feature_cols = feature_cols
        
        # Initialize models with default configurations
        self.models = self._initialize_models()
        
        # Setup output directories
        self.output_dir = output_dir
        self.models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize main log file
        self.main_log_file = os.path.join(self.output_dir, 'ml_run.txt')
        self._write_to_file(self.main_log_file, f"ML Framework Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", mode='w')

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize and return dictionary of models with their configurations."""
        return {
            # linear models
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            # 'enet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000),
            
            # tree based models
            # 'rf': RandomForestRegressor(
            #     n_estimators=400,          # plenty with subsampling
            #     max_depth=16,              # cap depth to keep trees compact
            #     min_samples_leaf=100,      # larger leaves reduce noise/overfit
            #     min_samples_split=200,     # pairs well with larger leaves
            #     max_samples=0.5,           # subsample 50% of rows per tree (huge speedup)
            #     n_jobs=-1,
            #     random_state=42
            # ),
            'xgb': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, 
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
            'lgbm': LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, 
                                num_leaves=31, n_jobs=-1),
            # 'catboost': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0),
            
            # neural network models
            # 'nn1': MLPRegressor(**nn_config['nn1_stable']),
            # 'nn2': MLPRegressor(**nn_config['nn2_deep']),
            # 'nn3': MLPRegressor(**nn_config['nn3_fast']),
        }
        
    def _write_to_file(self, filepath: str, content: str, mode: str = 'a') -> None:
        """Write content to a file."""
        with open(filepath, mode) as f:
            f.write(content + '\n')
            
    def _log_to_both(self, message: str, model_name: Optional[str] = None) -> None:
        """Log message to both console and file."""
        logger.info(message)
        if model_name:
            message = f"[{model_name}] {message}"
        self._write_to_file(self.main_log_file, message)

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics in a consistent format."""
        for metric, value in metrics.items():
            self._log_to_both(f"{prefix}- {metric.upper()}: {value:.4f}")

    def prepare_data(self) -> None:
        """Prepare the data for training by selecting features and target."""
        # Store feature names
        
        # Prepare features and target
        self.X = self.train_df[self.feature_cols]
        self.y = self.train_df[self.target_col]

        self.X_test = self.test_df[self.feature_cols]
        self.y_test = self.test_df[self.target_col]

        # winsorize X
        self.X = self.X.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
        self.X = self.scaler.fit_transform(self.X)

        self.X_test = self.X_test.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
        self.X_test = self.scaler.fit_transform(self.X_test)
        
        self._log_to_both("Data split and scaling completed:")
        self._log_to_both(f"- Training set size: {len(self.X)} samples")
        self._log_to_both(f"- Test set size: {len(self.X_test)} samples")
        self._log_to_both(f"- Number of features: {len(self.feature_cols)}")
        self._log_to_both(f"- Target variable: {self.target_col}")

    def _get_feature_importance(self, model: Any) -> Optional[np.ndarray]:
        """Get feature importance or coefficients from a model."""
        if hasattr(model, 'feature_importances_'):
            self._log_to_both("Feature importances: feature_importances_")
            return model.feature_importances_
        elif hasattr(model, 'get_score'):
            self._log_to_both("Feature importances: get_score")
            return model.get_score()
        elif hasattr(model, 'coef_'):
            self._log_to_both("Feature importances: coef_")
            coef = model.coef_
            return coef[0] if coef.ndim > 1 else coef
        return None

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all regression metrics for given predictions."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def _process_cv_results(self, cv_results: Dict[str, np.ndarray], scoring: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Process cross-validation results for all metrics."""
        model_metrics = {}
        for metric in scoring:
            scores_array = cv_results[f'test_{metric}']
            # Convert negative scores back to positive for MSE, RMSE, and MAE
            if metric in ['mse', 'rmse', 'mae']:
                scores_array = -scores_array
            
            model_metrics[metric] = {
                'mean': scores_array.mean(),
                'std': scores_array.std(),
                'scores': scores_array
            }
        return model_metrics

    def _log_model_metrics(self, model_metrics: Dict[str, Dict[str, float]], model_name: str) -> None:
        """Log model metrics in a consistent format."""
        self._log_to_both(f"Model results:", model_name)
        for metric_name, metric_data in model_metrics.items():
            self._log_to_both(
                f"- {metric_name.upper()}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}",
                model_name
            )
            self._log_to_both(f"- Individual fold scores: {metric_data['scores']}", model_name)

    def _save_feature_importance(self, model: Any, model_name: str) -> None:
        """Calculate and save feature importance for a model."""
        importance = self._get_feature_importance(model)
        if importance is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            })
            
            # Sort by absolute importance
            importance_df['abs_importance'] = np.abs(importance_df['importance'])
            importance_df = importance_df.sort_values('abs_importance', ascending=False).drop('abs_importance', axis=1)

            # Save to CSV
            importance_file = os.path.join(self.models_dir, f"{model_name}_feature_importance.csv")
            importance_df.to_csv(importance_file, index=False)

    def train_models(self) -> Dict[str, Dict[str, float]]:
        """Train multiple models and evaluate their performance using multiple metrics."""
        scores = {}
        self._log_to_both(f"Starting model training")
        
        for name, model in self.models.items():
            self._log_to_both(f"\nTraining {name} model...", name)
            
            # Perform cross-validation with multiple metrics at once
            model.fit(self.X, self.y)
            
            # Save feature importance
            self._save_feature_importance(model, name)
        
            model_metrics = self._calculate_metrics(self.y, model.predict(self.X))
            scores[name] = model_metrics

            # update in self.models the trained model
            self.models[name] = model

        # Save test metrics to CSV
        test_scores = []
        for model_name, metrics in scores.items():
            test_scores.append({
                'model': model_name,
                'r2': metrics['r2'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae']
            })
        # Save consolidated files
        pd.DataFrame(test_scores).to_csv(os.path.join(self.models_dir, 'scores.csv'), index=False)
                
        return test_scores
    
    def evaluate_model(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on the test set."""
        if not self.models:
            raise ValueError("No models specified.")
        
        self._log_to_both("\nEvaluating all models on test set...")
        test_metrics = {}
        
        for name, model in self.models.items():
            self._log_to_both(f"\nEvaluating {name} model...", name)
            
            # Make predictions
            y_pred = self.models[name].predict(self.X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred)
            test_metrics[name] = metrics

            # store the predictions in the dataframe of the test set
            self.test_df[f'y_pred_{name}'] = y_pred
            
            # Log metrics
            self._log_to_both("Test set metrics:", name)
            self._log_metrics(metrics)
        
        self.test_df['y_true'] = self.y_test
        
        # Save test metrics to CSV
        test_scores = []
        for model_name, metrics in test_metrics.items():
            test_scores.append({
                'model': model_name,
                'r2': metrics['r2'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae']
            })
        
        # Save to CSV
        pd.DataFrame(test_scores).to_csv(
            os.path.join(self.models_dir, 'oos_test_scores.csv'), 
            index=False
        )

        # Save the test set to CSV
        self.test_df.to_parquet(os.path.join(self.models_dir, 'oos_test_set.parquet'), index=False)
        
        return test_metrics