"""
Missing values management module - Simplified version for rapid delivery
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Union, List, Optional, Any, Dict
import logging

class MissingValueHandler:
    """Handles missing value imputation - Accelerated version"""
    
    def __init__(self, strategy: str = 'knn', k: int = 3):
        self.strategy = strategy
        self.k = k
        self.logger = logging.getLogger('MissingValueHandler')
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies imputation according to the chosen strategy"""
        self.logger.info(f"Imputing missing values with strategy: {self.strategy}")
        
        # Check if there are missing values
        missing_before = data.isnull().sum().sum()
        if missing_before == 0:
            self.logger.info("No missing values detected")
            return data
        
        self.logger.info(f"Missing values before imputation: {missing_before}")
        
        if self.strategy == 'knn':
            return self._knn_imputation(data)
        elif self.strategy == 'median':
            return self._median_imputation(data)
        elif self.strategy == 'mean':
            return self._mean_imputation(data)
        else:
            raise ValueError(f"Strategy '{self.strategy}' not supported")
    
    def _knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """KNN Imputation - Simplified version"""
        try:
            # Separate numeric and categorical columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns
            
            # Imputation for numeric columns
            if len(numeric_cols) > 0:
                numeric_data = data[numeric_cols].copy()
                imputer = KNNImputer(n_neighbors=min(self.k, len(numeric_data)))
                numeric_imputed = imputer.fit_transform(numeric_data)
                data[numeric_cols] = numeric_imputed
            
            # Imputation for categorical columns (mode)
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                    data[col] = data[col].fillna(mode_value)
            
            missing_after = data.isnull().sum().sum()
            self.logger.info(f"[OK] KNN imputation completed. Missing values after: {missing_after}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error during KNN imputation: {e}")
            # Fallback to median
            return self._median_imputation(data)
    
    def _median_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Median imputation - Simple fallback"""
        self.logger.info("Median imputation (fallback)")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Median for numeric
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
        
        # Mode for categorical
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_val)
        
        missing_after = data.isnull().sum().sum()
        self.logger.info(f"[OK] Median imputation completed. Missing values after: {missing_after}")
        
        return data
    
    def _mean_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mean imputation"""
        self.logger.info("Mean imputation")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Mean for numeric
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                mean_val = data[col].mean()
                data[col] = data[col].fillna(mean_val)
        
        # Mode for categorical
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_val)
        
        missing_after = data.isnull().sum().sum()
        self.logger.info(f"[OK] Mean imputation completed. Missing values after: {missing_after}")
        
        return data
    
    def filter_low_quality_features(self, data: pd.DataFrame, 
                                   threshold: float = 0.5) -> pd.DataFrame:
        """Filters features with too many missing values"""
        self.logger.info(f"Filtering features with >{threshold*100}% missing values")
        
        # Calculate percentage of missing values per column
        missing_percentages = data.isnull().sum() / len(data)
        
        # Identify columns to keep
        cols_to_keep = missing_percentages[missing_percentages <= threshold].index
        
        filtered_data = data[cols_to_keep].copy()
        
        removed_cols = len(data.columns) - len(cols_to_keep)
        self.logger.info(f"Features removed: {removed_cols}, Features kept: {len(cols_to_keep)}")
        
        return filtered_data
    
    def get_missing_value_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generates a report on missing values"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        report = {
            'total_missing_values': missing_counts.sum(),
            'total_missing_percentage': (missing_counts.sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
        
        return report

# Quick tests
if __name__ == "__main__":
    # Create test data
    test_data = pd.DataFrame({
        'gene1': [1, 2, np.nan, 4, 5],
        'gene2': [2, np.nan, 4, 5, 6],
        'gene3': [1, 2, 3, 4, 5],
        'category': ['A', 'B', np.nan, 'A', 'B']
    })
    
    print("=== MISSING VALUE HANDLER TEST ===")
    print("Original data:")
    print(test_data)
    print(f"Missing values: {test_data.isnull().sum().sum()}")
    
    # KNN Test
    handler = MissingValueHandler(strategy='knn', k=2)
    imputed_data = handler.fit_transform(test_data.copy())
    
    print("\nAfter KNN imputation:")
    print(imputed_data)
    print(f"Missing values after: {imputed_data.isnull().sum().sum()}")
    
    # Report
    report = handler.get_missing_value_report(test_data)
    print(f"\nReport: {report}")