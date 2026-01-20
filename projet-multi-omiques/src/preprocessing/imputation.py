
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging
from ..exceptions import PreprocessingError, MissingDataError

logger = logging.getLogger(__name__)


class MissingDataImputer:
    """Handle missing data imputation for multi-omics datasets."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the imputer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strategy = self.config.get('strategy', 'iterative')
        self.threshold = self.config.get('threshold', 0.3)  # Max missing percentage
        self.feature_specific = self.config.get('feature_specific', True)
        
        self.imputers = {}  # Store fitted imputers
        self.fitted_params = {}
        
        logger.info(f"Initialized MissingDataImputer with strategy: {self.strategy}")
    
    def fit_transform(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                     **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fit imputer and transform data.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            **kwargs: Additional parameters
            
        Returns:
            Imputed data
        """
        logger.info(f"Fitting imputer with strategy: {self.strategy}")
        
        if isinstance(data, dict):
            # Multi-omics data
            return self._fit_transform_multi_omics(data, **kwargs)
        else:
            # Single DataFrame
            return self._fit_transform_single(data, **kwargs)
    
    def transform(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                 **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Transform data using fitted imputer.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            **kwargs: Additional parameters
            
        Returns:
            Imputed data
        """
        logger.info("Transforming data with fitted imputer")
        
        if isinstance(data, dict):
            return self._transform_multi_omics(data, **kwargs)
        else:
            return self._transform_single(data, **kwargs)
    
    def _fit_transform_single(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform single DataFrame."""
        # Check missing data percentage
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        
        if missing_pct > self.threshold:
            raise MissingDataError(
                f"Too much missing data: {missing_pct:.2%} > {self.threshold:.2%}"
            )
        
        # Remove features with too many missing values
        feature_missing = df.isnull().sum() / len(df)
        features_to_keep = feature_missing[feature_missing <= self.threshold].index
        
        logger.info(f"Keeping {len(features_to_keep)}/{len(df.columns)} features")
        
        df_clean = df[features_to_keep].copy()
        
        # Apply imputation strategy
        if self.strategy == 'simple':
            imputed_df = self._simple_impute(df_clean, fit=True, **kwargs)
        elif self.strategy == 'iterative':
            imputed_df = self._iterative_impute(df_clean, fit=True, **kwargs)
        elif self.strategy == 'knn':
            imputed_df = self._knn_impute(df_clean, fit=True, **kwargs)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.strategy}")
        
        return imputed_df
    
    def _fit_transform_multi_omics(self, data_dict: Dict[str, pd.DataFrame], 
                                  **kwargs) -> Dict[str, pd.DataFrame]:
        """Fit and transform multi-omics data."""
        imputed_data = {}
        
        for omics_type, df in data_dict.items():
            logger.info(f"Processing {omics_type} data")
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {omics_type}, skipping")
                continue
            
            try:
                # Apply omics-specific imputation parameters
                omics_config = self.config.get(omics_type, {})
                original_strategy = self.strategy
                
                if 'strategy' in omics_config:
                    self.strategy = omics_config['strategy']
                
                imputed_df = self._fit_transform_single(df, **kwargs)
                imputed_data[omics_type] = imputed_df
                
                # Restore original strategy
                self.strategy = original_strategy
                
            except Exception as e:
                logger.error(f"Failed to impute {omics_type} data: {e}")
                # Keep original data if imputation fails
                imputed_data[omics_type] = df
        
        return imputed_data
    
    def _transform_single(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform single DataFrame using fitted imputer."""
        if not self.imputers:
            raise PreprocessingError("Imputer not fitted. Call fit_transform first.")
        
        # Apply same feature selection as in fit
        feature_missing = df.isnull().sum() / len(df)
        features_to_keep = feature_missing[feature_missing <= self.threshold].index
        
        df_clean = df[features_to_keep].copy()
        
        # Apply fitted imputation
        if self.strategy == 'simple':
            imputed_df = self._simple_impute(df_clean, fit=False, **kwargs)
        elif self.strategy == 'iterative':
            imputed_df = self._iterative_impute(df_clean, fit=False, **kwargs)
        elif self.strategy == 'knn':
            imputed_df = self._knn_impute(df_clean, fit=False, **kwargs)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.strategy}")
        
        return imputed_df
    
    def _transform_multi_omics(self, data_dict: Dict[str, pd.DataFrame], 
                              **kwargs) -> Dict[str, pd.DataFrame]:
        """Transform multi-omics data using fitted imputers."""
        imputed_data = {}
        
        for omics_type, df in data_dict.items():
            logger.info(f"Transforming {omics_type} data")
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {omics_type}, skipping")
                continue
            
            try:
                # Check if we have a fitted imputer for this omics type
                if omics_type in self.imputers:
                    imputed_df = self._transform_single(df, **kwargs)
                    imputed_data[omics_type] = imputed_df
                else:
                    logger.warning(f"No fitted imputer for {omics_type}, keeping original data")
                    imputed_data[omics_type] = df
                    
            except Exception as e:
                logger.error(f"Failed to transform {omics_type} data: {e}")
                imputed_data[omics_type] = df
        
        return imputed_data
    
    def _simple_impute(self, df: pd.DataFrame, fit: bool = True, 
                      strategy: str = 'median') -> pd.DataFrame:
        """Apply simple imputation strategies."""
        logger.info(f"Applying simple imputation (strategy: {strategy})")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        imputed_df = df.copy()
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            if fit:
                self.imputers['numeric'] = SimpleImputer(strategy=strategy)
                imputed_numeric = self.imputers['numeric'].fit_transform(df[numeric_cols])
            else:
                imputed_numeric = self.imputers['numeric'].transform(df[numeric_cols])
            
            imputed_df[numeric_cols] = imputed_numeric
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            if fit:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                imputed_categorical = self.imputers['categorical'].fit_transform(df[categorical_cols])
            else:
                imputed_categorical = self.imputers['categorical'].transform(df[categorical_cols])
            
            imputed_df[categorical_cols] = imputed_categorical
        
        return imputed_df
    
    def _iterative_impute(self, df: pd.DataFrame, fit: bool = True, 
                         max_iter: int = 10, random_state: int = 42) -> pd.DataFrame:
        """Apply iterative imputation using Random Forest."""
        logger.info("Applying iterative imputation")
        
        # Use only numeric data for iterative imputation
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns for iterative imputation")
            return df
        
        # Standardize data
        if fit:
            self.fitted_params['scaler'] = StandardScaler()
            scaled_data = self.fitted_params['scaler'].fit_transform(numeric_df)
        else:
            scaled_data = self.fitted_params['scaler'].transform(numeric_df)
        
        # Apply iterative imputer
        if fit:
            estimator = RandomForestRegressor(n_estimators=10, random_state=random_state)
            self.imputers['iterative'] = IterativeImputer(
                estimator=estimator,
                max_iter=max_iter,
                random_state=random_state,
                verbose=1
            )
            imputed_scaled = self.imputers['iterative'].fit_transform(scaled_data)
        else:
            imputed_scaled = self.imputers['iterative'].transform(scaled_data)
        
        # Reverse scaling
        imputed_numeric = self.fitted_params['scaler'].inverse_transform(imputed_scaled)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df[numeric_df.columns] = imputed_numeric
        
        return result_df
    
    def _knn_impute(self, df: pd.DataFrame, fit: bool = True, 
                   n_neighbors: int = 5, weights: str = 'distance') -> pd.DataFrame:
        """Apply KNN imputation."""
        logger.info(f"Applying KNN imputation (k={n_neighbors})")
        
        # Use only numeric data for KNN imputation
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns for KNN imputation")
            return df
        
        if fit:
            self.imputers['knn'] = KNNImputer(
                n_neighbors=n_neighbors,
                weights=weights
            )
            imputed_numeric = self.imputers['knn'].fit_transform(numeric_df)
        else:
            imputed_numeric = self.imputers['knn'].transform(numeric_df)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df[numeric_df.columns] = imputed_numeric
        
        return result_df
    
    def get_missing_data_summary(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Get summary of missing data patterns.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            
        Returns:
            Missing data summary
        """
        summary = {}
        
        if isinstance(data, dict):
            for omics_type, df in data.items():
                summary[omics_type] = self._get_single_missing_summary(df)
        else:
            summary['single_dataset'] = self._get_single_missing_summary(data)
        
        return summary
    
    def _get_single_missing_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get missing data summary for single DataFrame."""
        total_values = df.shape[0] * df.shape[1]
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / total_values) * 100 if total_values > 0 else 0
        
        # Per-column missing data
        col_missing = df.isnull().sum()
        col_missing_pct = (col_missing / len(df)) * 100
        
        # Rows with missing data
        rows_with_missing = df.isnull().any(axis=1).sum()
        rows_missing_pct = (rows_with_missing / len(df)) * 100
        
        return {
            'total_values': total_values,
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'columns_with_missing': col_missing[col_missing > 0].to_dict(),
            'columns_missing_percentage': col_missing_pct[col_missing > 0].to_dict(),
            'rows_with_missing': rows_with_missing,
            'rows_missing_percentage': rows_missing_pct,
            'complete_cases': len(df) - rows_with_missing,
            'complete_cases_percentage': 100 - rows_missing_pct
        }