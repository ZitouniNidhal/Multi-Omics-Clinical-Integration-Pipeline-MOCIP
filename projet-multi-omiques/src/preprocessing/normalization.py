"""Normalization strategies for different omics data types."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import logging
from ..exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Handle data normalization for multi-omics datasets."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the normalizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scalers = {}  # Store fitted scalers
        self.fitted_params = {}
        
        logger.info("Initialized DataNormalizer")
    
    def fit_transform(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                     data_type: str = None, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fit normalizer and transform data.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            data_type: Type of data (gene_expression, proteomics, metabolomics)
            **kwargs: Additional parameters
            
        Returns:
            Normalized data
        """
        logger.info(f"Fitting normalizer for data type: {data_type}")
        
        if isinstance(data, dict):
            # Multi-omics data
            return self._fit_transform_multi_omics(data, **kwargs)
        else:
            # Single DataFrame
            return self._fit_transform_single(data, data_type, **kwargs)
    
    def transform(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                 data_type: str = None, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Transform data using fitted normalizer.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            data_type: Type of data
            **kwargs: Additional parameters
            
        Returns:
            Normalized data
        """
        logger.info(f"Transforming data with fitted normalizer")
        
        if isinstance(data, dict):
            return self._transform_multi_omics(data, **kwargs)
        else:
            return self._transform_single(data, data_type, **kwargs)
    
    def _fit_transform_single(self, df: pd.DataFrame, data_type: str = None, 
                             **kwargs) -> pd.DataFrame:
        """Fit and transform single DataFrame."""
        if df.empty:
            logger.warning("Empty DataFrame, returning unchanged")
            return df
        
        # Determine normalization method based on data type
        if data_type == 'gene_expression':
            normalized_df = self._normalize_gene_expression(df, fit=True, **kwargs)
        elif data_type == 'proteomics':
            normalized_df = self._normalize_proteomics(df, fit=True, **kwargs)
        elif data_type == 'metabolomics':
            normalized_df = self._normalize_metabolomics(df, fit=True, **kwargs)
        elif data_type == 'clinical':
            normalized_df = self._normalize_clinical(df, fit=True, **kwargs)
        else:
            # Default normalization
            normalized_df = self._normalize_generic(df, fit=True, **kwargs)
        
        return normalized_df
    
    def _fit_transform_multi_omics(self, data_dict: Dict[str, pd.DataFrame], 
                                  **kwargs) -> Dict[str, pd.DataFrame]:
        """Fit and transform multi-omics data."""
        normalized_data = {}
        
        for omics_type, df in data_dict.items():
            logger.info(f"Normalizing {omics_type} data")
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {omics_type}, skipping")
                continue
            
            try:
                # Apply omics-specific normalization
                normalized_df = self._fit_transform_single(df, omics_type, **kwargs)
                normalized_data[omics_type] = normalized_df
                
            except Exception as e:
                logger.error(f"Failed to normalize {omics_type} data: {e}")
                # Keep original data if normalization fails
                normalized_data[omics_type] = df
        
        return normalized_data
    
    def _transform_single(self, df: pd.DataFrame, data_type: str = None, 
                         **kwargs) -> pd.DataFrame:
        """Transform single DataFrame using fitted normalizer."""
        if not self.scalers and not self.fitted_params:
            logger.warning("Normalizer not fitted. Returning original data.")
            return df
        
        # Determine normalization method based on data type
        if data_type == 'gene_expression':
            normalized_df = self._normalize_gene_expression(df, fit=False, **kwargs)
        elif data_type == 'proteomics':
            normalized_df = self._normalize_proteomics(df, fit=False, **kwargs)
        elif data_type == 'metabolomics':
            normalized_df = self._normalize_metabolomics(df, fit=False, **kwargs)
        elif data_type == 'clinical':
            normalized_df = self._normalize_clinical(df, fit=False, **kwargs)
        else:
            # Default normalization
            normalized_df = self._normalize_generic(df, fit=False, **kwargs)
        
        return normalized_df
    
    def _transform_multi_omics(self, data_dict: Dict[str, pd.DataFrame], 
                              **kwargs) -> Dict[str, pd.DataFrame]:
        """Transform multi-omics data using fitted normalizers."""
        normalized_data = {}
        
        for omics_type, df in data_dict.items():
            logger.info(f"Transforming {omics_type} data")
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {omics_type}, skipping")
                continue
            
            try:
                # Check if we have a fitted normalizer for this omics type
                if omics_type in self.scalers or omics_type in self.fitted_params:
                    normalized_df = self._transform_single(df, omics_type, **kwargs)
                    normalized_data[omics_type] = normalized_df
                else:
                    logger.warning(f"No fitted normalizer for {omics_type}, keeping original data")
                    normalized_data[omics_type] = df
                    
            except Exception as e:
                logger.error(f"Failed to transform {omics_type} data: {e}")
                normalized_data[omics_type] = df
        
        return normalized_data
    
    def _normalize_gene_expression(self, df: pd.DataFrame, fit: bool = True, 
                                  method: str = 'tpm', **kwargs) -> pd.DataFrame:
        """Normalize gene expression data."""
        logger.info(f"Normalizing gene expression data (method: {method})")
        
        if method == 'tpm':
            return self._tpm_normalization(df, fit=fit, **kwargs)
        elif method == 'fpkm':
            return self._fpkm_normalization(df, fit=fit, **kwargs)
        elif method == 'deseq2_size_factors':
            return self._deseq2_normalization(df, fit=fit, **kwargs)
        elif method == 'quantile':
            return self._quantile_normalization(df, fit=fit, **kwargs)
        elif method == 'standard':
            return self._standard_scaling(df, fit=fit, **kwargs)
        else:
            raise ValueError(f"Unknown gene expression normalization method: {method}")
    
    def _normalize_proteomics(self, df: pd.DataFrame, fit: bool = True, 
                             method: str = 'median_centering', **kwargs) -> pd.DataFrame:
        """Normalize proteomics data."""
        logger.info(f"Normalizing proteomics data (method: {method})")
        
        if method == 'median_centering':
            return self._median_centering(df, fit=fit, **kwargs)
        elif method == 'quantile':
            return self._quantile_normalization(df, fit=fit, **kwargs)
        elif method == 'standard':
            return self._standard_scaling(df, fit=fit, **kwargs)
        elif method == 'robust':
            return self._robust_scaling(df, fit=fit, **kwargs)
        else:
            raise ValueError(f"Unknown proteomics normalization method: {method}")
    
    def _normalize_metabolomics(self, df: pd.DataFrame, fit: bool = True, 
                               method: str = 'pareto_scaling', **kwargs) -> pd.DataFrame:
        """Normalize metabolomics data."""
        logger.info(f"Normalizing metabolomics data (method: {method})")
        
        if method == 'pareto_scaling':
            return self._pareto_scaling(df, fit=fit, **kwargs)
        elif method == 'auto_scaling':
            return self._auto_scaling(df, fit=fit, **kwargs)
        elif method == 'range_scaling':
            return self._range_scaling(df, fit=fit, **kwargs)
        elif method == 'log_transform':
            return self._log_transformation(df, fit=fit, **kwargs)
        else:
            raise ValueError(f"Unknown metabolomics normalization method: {method}")
    
    def _normalize_clinical(self, df: pd.DataFrame, fit: bool = True, 
                           method: str = 'standard', **kwargs) -> pd.DataFrame:
        """Normalize clinical data."""
        logger.info(f"Normalizing clinical data (method: {method})")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        normalized_df = df.copy()
        
        # Normalize numeric columns
        if len(numeric_cols) > 0:
            if method == 'standard':
                numeric_normalized = self._standard_scaling(df[numeric_cols], fit=fit, **kwargs)
            elif method == 'minmax':
                numeric_normalized = self._minmax_scaling(df[numeric_cols], fit=fit, **kwargs)
            elif method == 'robust':
                numeric_normalized = self._robust_scaling(df[numeric_cols], fit=fit, **kwargs)
            else:
                raise ValueError(f"Unknown clinical normalization method: {method}")
            
            normalized_df[numeric_cols] = numeric_normalized
        
        # Categorical columns remain unchanged
        if len(categorical_cols) > 0:
            logger.info(f"Keeping {len(categorical_cols)} categorical columns unchanged")
        
        return normalized_df
    
    def _normalize_generic(self, df: pd.DataFrame, fit: bool = True, 
                          method: str = 'standard', **kwargs) -> pd.DataFrame:
        """Generic normalization for unknown data types."""
        logger.info(f"Applying generic normalization (method: {method})")
        
        if method == 'standard':
            return self._standard_scaling(df, fit=fit, **kwargs)
        elif method == 'minmax':
            return self._minmax_scaling(df, fit=fit, **kwargs)
        elif method == 'robust':
            return self._robust_scaling(df, fit=fit, **kwargs)
        else:
            raise ValueError(f"Unknown generic normalization method: {method}")
    
    def _tpm_normalization(self, df: pd.DataFrame, fit: bool = True, 
                          gene_lengths: Optional[pd.Series] = None) -> pd.DataFrame:
        """TPM (Transcripts Per Million) normalization."""
        logger.info("Applying TPM normalization")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for TPM normalization")
            return df
        
        # If gene lengths provided, adjust for gene length
        if gene_lengths is not None:
            # Divide by gene length (in kb)
            gene_lengths_kb = gene_lengths / 1000
            df_normalized = df_numeric.div(gene_lengths_kb, axis=0)
        else:
            df_normalized = df_numeric.copy()
        
        # Calculate TPM
        # 1. Divide each count by the total count of its sample
        per_million_scaling = df_normalized.sum(axis=0) / 1e6
        tpm_matrix = df_normalized.div(per_million_scaling, axis=1)
        
        # Fill any infinite values
        tpm_matrix = tpm_matrix.replace([np.inf, -np.inf], 0)
        
        logger.info(f"TPM normalization complete. Shape: {tpm_matrix.shape}")
        
        return tpm_matrix
    
    def _fpkm_normalization(self, df: pd.DataFrame, fit: bool = True, 
                           gene_lengths: Optional[pd.Series] = None) -> pd.DataFrame:
        """FPKM (Fragments Per Kilobase Million) normalization."""
        logger.info("Applying FPKM normalization")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for FPKM normalization")
            return df
        
        if gene_lengths is None:
            logger.warning("Gene lengths not provided, using TPM instead")
            return self._tpm_normalization(df, fit=fit)
        
        # Calculate FPKM
        # 1. Divide by gene length (in kb)
        gene_lengths_kb = gene_lengths / 1000
        df_length_normalized = df_numeric.div(gene_lengths_kb, axis=0)
        
        # 2. Divide by total counts per sample (in millions)
        per_million_scaling = df_numeric.sum(axis=0) / 1e6
        fpkm_matrix = df_length_normalized.div(per_million_scaling, axis=1)
        
        # Fill any infinite values
        fpkm_matrix = fpkm_matrix.replace([np.inf, -np.inf], 0)
        
        logger.info(f"FPKM normalization complete. Shape: {fpkm_matrix.shape}")
        
        return fpkm_matrix
    
    def _deseq2_normalization(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """DESeq2 size factors normalization."""
        logger.info("Applying DESeq2 size factors normalization")
        
        # Ensure data is numeric and positive
        df_numeric = df.select_dtypes(include=[np.number])
        df_numeric = df_numeric.clip(lower=0)  # Ensure non-negative
        
        if df_numeric.empty:
            logger.warning("No numeric data for DESeq2 normalization")
            return df
        
        # Calculate geometric mean for each gene (row)
        geometric_means = np.exp(np.log1p(df_numeric).mean(axis=1))
        
        # Remove genes with zero geometric mean
        valid_genes = geometric_means > 0
        df_valid = df_numeric.loc[valid_genes]
        geometric_means = geometric_means[valid_genes]
        
        # Calculate size factors for each sample
        size_factors = []
        for sample in df_valid.columns:
            ratios = df_valid[sample] / geometric_means
            # Use median of ratios (excluding zeros and infinities)
            valid_ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            if len(valid_ratios) > 0:
                size_factor = np.median(valid_ratios)
            else:
                size_factor = 1.0
            size_factors.append(size_factor)
        
        size_factors = pd.Series(size_factors, index=df_valid.columns)
        
        # Apply size factors
        normalized_df = df_numeric.div(size_factors, axis=1)
        
        # Store size factors for later use
        if fit:
            self.fitted_params['deseq2_size_factors'] = size_factors
        
        logger.info(f"DESeq2 normalization complete. Shape: {normalized_df.shape}")
        
        return normalized_df
    
    def _quantile_normalization(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Quantile normalization."""
        logger.info("Applying quantile normalization")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for quantile normalization")
            return df
        
        # Sort each column
        sorted_data = np.sort(df_numeric.values, axis=0)
        
        # Calculate mean for each rank
        rank_means = np.mean(sorted_data, axis=1)
        
        # Create normalized data
        normalized_data = df_numeric.copy()
        
        for col in df_numeric.columns:
            # Get ranks of original data
            ranks = df_numeric[col].rank(method='average', na_option='keep')
            
            # Replace values with rank means
            for i, rank in enumerate(ranks):
                if not pd.isna(rank):
                    rank_idx = int(rank) - 1
                    if 0 <= rank_idx < len(rank_means):
                        normalized_data.iloc[i, normalized_data.columns.get_loc(col)] = rank_means[rank_idx]
        
        logger.info(f"Quantile normalization complete. Shape: {normalized_data.shape}")
        
        return normalized_data
    
    def _standard_scaling(self, df: pd.DataFrame, fit: bool = True, 
                         with_mean: bool = True, with_std: bool = True) -> pd.DataFrame:
        """Standard scaling (z-score normalization)."""
        logger.info("Applying standard scaling")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for standard scaling")
            return df
        
        scaler_key = 'standard_scaler'
        
        if fit:
            self.scalers[scaler_key] = StandardScaler(with_mean=with_mean, with_std=with_std)
            scaled_data = self.scalers[scaler_key].fit_transform(df_numeric)
        else:
            if scaler_key not in self.scalers:
                logger.warning("Standard scaler not fitted, fitting now")
                self.scalers[scaler_key] = StandardScaler(with_mean=with_mean, with_std=with_std)
                scaled_data = self.scalers[scaler_key].fit_transform(df_numeric)
            else:
                scaled_data = self.scalers[scaler_key].transform(df_numeric)
        
        # Create result DataFrame
        scaled_df = pd.DataFrame(
            scaled_data,
            index=df_numeric.index,
            columns=df_numeric.columns
        )
        
        logger.info(f"Standard scaling complete. Shape: {scaled_df.shape}")
        
        return scaled_df
    
    def _minmax_scaling(self, df: pd.DataFrame, fit: bool = True, 
                       feature_range: tuple = (0, 1)) -> pd.DataFrame:
        """Min-max scaling."""
        logger.info("Applying min-max scaling")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for min-max scaling")
            return df
        
        scaler_key = 'minmax_scaler'
        
        if fit:
            self.scalers[scaler_key] = MinMaxScaler(feature_range=feature_range)
            scaled_data = self.scalers[scaler_key].fit_transform(df_numeric)
        else:
            if scaler_key not in self.scalers:
                logger.warning("MinMax scaler not fitted, fitting now")
                self.scalers[scaler_key] = MinMaxScaler(feature_range=feature_range)
                scaled_data = self.scalers[scaler_key].fit_transform(df_numeric)
            else:
                scaled_data = self.scalers[scaler_key].transform(df_numeric)
        
        # Create result DataFrame
        scaled_df = pd.DataFrame(
            scaled_data,
            index=df_numeric.index,
            columns=df_numeric.columns
        )
        
        logger.info(f"Min-max scaling complete. Shape: {scaled_df.shape}")
        
        return scaled_df
    
    def _robust_scaling(self, df: pd.DataFrame, fit: bool = True, 
                       with_centering: bool = True, with_scaling: bool = True) -> pd.DataFrame:
        """Robust scaling using median and IQR."""
        logger.info("Applying robust scaling")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for robust scaling")
            return df
        
        scaler_key = 'robust_scaler'
        
        if fit:
            self.scalers[scaler_key] = RobustScaler(
                with_centering=with_centering,
                with_scaling=with_scaling
            )
            scaled_data = self.scalers[scaler_key].fit_transform(df_numeric)
        else:
            if scaler_key not in self.scalers:
                logger.warning("Robust scaler not fitted, fitting now")
                self.scalers[scaler_key] = RobustScaler(
                    with_centering=with_centering,
                    with_scaling=with_scaling
                )
                scaled_data = self.scalers[scaler_key].fit_transform(df_numeric)
            else:
                scaled_data = self.scalers[scaler_key].transform(df_numeric)
        
        # Create result DataFrame
        scaled_df = pd.DataFrame(
            scaled_data,
            index=df_numeric.index,
            columns=df_numeric.columns
        )
        
        logger.info(f"Robust scaling complete. Shape: {scaled_df.shape}")
        
        return scaled_df
    
    def _median_centering(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Median centering for proteomics data."""
        logger.info("Applying median centering")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for median centering")
            return df
        
        if fit:
            # Calculate medians
            self.fitted_params['medians'] = df_numeric.median()
        
        # Apply median centering
        centered_df = df_numeric.sub(self.fitted_params['medians'], axis=1)
        
        logger.info(f"Median centering complete. Shape: {centered_df.shape}")
        
        return centered_df
    
    def _pareto_scaling(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Pareto scaling for metabolomics data."""
        logger.info("Applying Pareto scaling")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for Pareto scaling")
            return df
        
        if fit:
            # Calculate means and standard deviations
            self.fitted_params['means'] = df_numeric.mean()
            self.fitted_params['stds'] = df_numeric.std()
        
        # Apply Pareto scaling: (x - mean) / sqrt(std)
        scaled_df = df_numeric.sub(self.fitted_params['means'], axis=1)
        pareto_divisor = np.sqrt(self.fitted_params['stds'])
        pareto_divisor = pareto_divisor.replace(0, 1)  # Avoid division by zero
        scaled_df = scaled_df.div(pareto_divisor, axis=1)
        
        logger.info(f"Pareto scaling complete. Shape: {scaled_df.shape}")
        
        return scaled_df
    
    def _auto_scaling(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Auto scaling (standard scaling) for metabolomics."""
        logger.info("Applying auto scaling")
        
        # Auto scaling is the same as standard scaling
        return self._standard_scaling(df, fit=fit)
    
    def _range_scaling(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Range scaling for metabolomics."""
        logger.info("Applying range scaling")
        
        # Ensure data is numeric
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            logger.warning("No numeric data for range scaling")
            return df
        
        if fit:
            # Calculate min and max
            self.fitted_params['mins'] = df_numeric.min()
            self.fitted_params['maxs'] = df_numeric.max()
        
        # Apply range scaling: (x - min) / (max - min)
        ranges = self.fitted_params['maxs'] - self.fitted_params['mins']
        ranges = ranges.replace(0, 1)  # Avoid division by zero
        
        scaled_df = df_numeric.sub(self.fitted_params['mins'], axis=1)
        scaled_df = scaled_df.div(ranges, axis=1)
        
        logger.info(f"Range scaling complete. Shape: {scaled_df.shape}")
        
        return scaled_df
    
    def _log_transformation(self, df: pd.DataFrame, fit: bool = True, 
                           base: int = 2, offset: float = 1.0) -> pd.DataFrame:
        """Log transformation for metabolomics data."""
        logger.info(f"Applying log transformation (base: {base}, offset: {offset})")
        
        # Ensure data is numeric and positive
        df_numeric = df.select_dtypes(include=[np.number])
        df_numeric = df_numeric.clip(lower=0)  # Ensure non-negative
        
        if df_numeric.empty:
            logger.warning("No numeric data for log transformation")
            return df
        
        # Add offset to avoid log(0)
        df_offset = df_numeric + offset
        
        # Apply log transformation
        if base == 2:
            log_df = np.log2(df_offset)
        elif base == 10:
            log_df = np.log10(df_offset)
        elif base == np.e:
            log_df = np.log(df_offset)
        else:
            log_df = np.log(df_offset) / np.log(base)
        
        logger.info(f"Log transformation complete. Shape: {log_df.shape}")
        
        return log_df