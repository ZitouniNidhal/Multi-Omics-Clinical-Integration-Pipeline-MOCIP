"""Multi-omics data fusion strategies."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging
from ..exceptions import IntegrationError

logger = logging.getLogger(__name__)


class DataFusion:
    """Multi-omics data fusion using different integration strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data fusion.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.integration_methods = self.config.get('methods', ['early_integration'])
        self.scaling_method = self.config.get('scaling_method', 'standard')
        self.feature_selection = self.config.get('feature_selection', True)
        self.max_features_per_omics = self.config.get('max_features_per_omics', 1000)
        
        self.fitted_scalers = {}
        self.selected_features = {}
        
        logger.info(f"Initialized DataFusion with methods: {self.integration_methods}")
    
    def integrate(self, omics_data: Dict[str, pd.DataFrame], 
                  clinical_data: Optional[pd.DataFrame] = None,
                  sample_id_column: str = 'sample_id',
                  **kwargs) -> Dict[str, Any]:
        """
        Integrate multi-omics data using specified methods.
        
        Args:
            omics_data: Dictionary of omics DataFrames
            clinical_data: Optional clinical data
            sample_id_column: Column name for sample IDs
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing integrated data and metadata
        """
        logger.info(f"Integrating {len(omics_data)} omics datasets")
        
        # Align samples across datasets
        aligned_data = self._align_samples(omics_data, sample_id_column)
        
        if not aligned_data:
            raise IntegrationError("No common samples found across omics datasets")
        
        integration_results = {
            'aligned_data': aligned_data,
            'integration_methods': {},
            'sample_mapping': {},
            'quality_metrics': {}
        }
        
        # Apply each integration method
        for method in self.integration_methods:
            logger.info(f"Applying integration method: {method}")
            
            try:
                if method == 'early_integration':
                    integrated_data = self._early_integration(aligned_data, clinical_data, **kwargs)
                elif method == 'late_integration':
                    integrated_data = self._late_integration(aligned_data, clinical_data, **kwargs)
                elif method == 'intermediate_integration':
                    integrated_data = self._intermediate_integration(aligned_data, clinical_data, **kwargs)
                else:
                    logger.warning(f"Unknown integration method: {method}")
                    continue
                
                integration_results['integration_methods'][method] = integrated_data
                
            except Exception as e:
                logger.error(f"Integration method {method} failed: {e}")
                integration_results['integration_methods'][method] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate integration quality metrics
        integration_results['quality_metrics'] = self._calculate_integration_metrics(integration_results)
        
        logger.info("Data integration complete")
        
        return integration_results
    
    def _align_samples(self, omics_data: Dict[str, pd.DataFrame], 
                      sample_id_column: str) -> Dict[str, pd.DataFrame]:
        """Align samples across different omics datasets."""
        logger.info("Aligning samples across omics datasets")
        
        # Find common samples
        common_samples = None
        
        for omics_type, df in omics_data.items():
            if df.empty:
                logger.warning(f"Empty DataFrame for {omics_type}, skipping")
                continue
            
            if sample_id_column not in df.columns:
                logger.warning(f"Sample ID column '{sample_id_column}' not found in {omics_type}")
                continue
            
            current_samples = set(df[sample_id_column].values)
            
            if common_samples is None:
                common_samples = current_samples
            else:
                common_samples = common_samples.intersection(current_samples)
            
            logger.info(f"{omics_type}: {len(current_samples)} samples")
        
        if not common_samples:
            logger.error("No common samples found across omics datasets")
            return {}
        
        common_samples = sorted(list(common_samples))
        logger.info(f"Found {len(common_samples)} common samples")
        
        # Filter each dataset to common samples
        aligned_data = {}
        
        for omics_type, df in omics_data.items():
            if df.empty or sample_id_column not in df.columns:
                continue
            
            # Filter to common samples
            filtered_df = df[df[sample_id_column].isin(common_samples)].copy()
            
            # Sort by sample ID for consistency
            filtered_df = filtered_df.sort_values(sample_id_column)
            
            aligned_data[omics_type] = filtered_df
            
            logger.info(f"Aligned {omics_type}: {len(filtered_df)} samples")
        
        return aligned_data
    
    def _early_integration(self, omics_data: Dict[str, pd.DataFrame], 
                          clinical_data: Optional[pd.DataFrame] = None,
                          **kwargs) -> Dict[str, Any]:
        """Early integration - concatenate features from all omics."""
        logger.info("Performing early integration")
        
        # Prepare omics data
        omics_features = []
        sample_ids = None
        
        for omics_type, df in omics_data.items():
            if df.empty:
                continue
            
            # Separate sample IDs from features
            if 'sample_id' in df.columns:
                current_sample_ids = df['sample_id'].values
                feature_df = df.drop(columns=['sample_id'])
            else:
                current_sample_ids = df.index.values
                feature_df = df
            
            # Ensure sample IDs are consistent
            if sample_ids is None:
                sample_ids = current_sample_ids
            elif not np.array_equal(sample_ids, current_sample_ids):
                raise IntegrationError(f"Inconsistent sample IDs in {omics_type}")
            
            # Add prefix to feature names to avoid conflicts
            feature_df = feature_df.add_prefix(f"{omics_type}_")
            
            # Scale features
            scaled_features = self._scale_features(feature_df, omics_type, fit=True)
            
            omics_features.append(scaled_features)
        
        if not omics_features:
            raise IntegrationError("No features available for integration")
        
        # Concatenate features
        integrated_features = pd.concat(omics_features, axis=1)
        
        # Add clinical data if provided
        if clinical_data is not None and not clinical_data.empty:
            clinical_features = self._prepare_clinical_data(clinical_data, sample_ids)
            integrated_features = pd.concat([integrated_features, clinical_features], axis=1)
        
        # Create final integrated dataset
        integrated_data = pd.DataFrame({
            'sample_id': sample_ids
        })
        
        integrated_data = pd.concat([integrated_data, integrated_features], axis=1)
        
        result = {
            'integrated_data': integrated_data,
            'n_samples': len(sample_ids),
            'n_features': len(integrated_features.columns),
            'omics_types': list(omics_data.keys()),
            'method': 'early_integration',
            'feature_names': list(integrated_features.columns),
            'scaling_info': self._get_scaling_info()
        }
        
        logger.info(f"Early integration complete: {result['n_samples']} samples, {result['n_features']} features")
        
        return result
    
    def _late_integration(self, omics_data: Dict[str, pd.DataFrame], 
                         clinical_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> Dict[str, Any]:
        """Late integration - combine predictions from individual omics."""
        logger.info("Performing late integration")
        
        # Process each omics dataset separately
        processed_omics = {}
        sample_ids = None
        
        for omics_type, df in omics_data.items():
            if df.empty:
                continue
            
            # Extract features and sample IDs
            if 'sample_id' in df.columns:
                current_sample_ids = df['sample_id'].values
                feature_df = df.drop(columns=['sample_id'])
            else:
                current_sample_ids = df.index.values
                feature_df = df
            
            # Ensure sample IDs are consistent
            if sample_ids is None:
                sample_ids = current_sample_ids
            elif not np.array_equal(sample_ids, current_sample_ids):
                raise IntegrationError(f"Inconsistent sample IDs in {omics_type}")
            
            # Scale features
            scaled_features = self._scale_features(feature_df, omics_type, fit=True)
            
            processed_omics[omics_type] = {
                'features': scaled_features,
                'sample_ids': current_sample_ids
            }
        
        if not processed_omics:
            raise IntegrationError("No omics data available for integration")
        
        # Prepare clinical data if provided
        clinical_features = None
        if clinical_data is not None and not clinical_data.empty:
            clinical_features = self._prepare_clinical_data(clinical_data, sample_ids)
        
        # Create integrated dataset structure
        integrated_data = pd.DataFrame({
            'sample_id': sample_ids
        })
        
        result = {
            'integrated_data': integrated_data,
            'individual_omics': processed_omics,
            'clinical_data': clinical_features,
            'n_samples': len(sample_ids),
            'omics_types': list(processed_omics.keys()),
            'method': 'late_integration',
            'scaling_info': self._get_scaling_info()
        }
        
        logger.info(f"Late integration complete: {result['n_samples']} samples, {len(processed_omics)} omics types")
        
        return result
    
    def _intermediate_integration(self, omics_data: Dict[str, pd.DataFrame], 
                                 clinical_data: Optional[pd.DataFrame] = None,
                                 **kwargs) -> Dict[str, Any]:
        """Intermediate integration - combine at feature level with dimensionality reduction."""
        logger.info("Performing intermediate integration")
        
        # Apply dimensionality reduction to each omics type
        reduced_omics = {}
        sample_ids = None
        
        for omics_type, df in omics_data.items():
            if df.empty:
                continue
            
            # Extract features and sample IDs
            if 'sample_id' in df.columns:
                current_sample_ids = df['sample_id'].values
                feature_df = df.drop(columns=['sample_id'])
            else:
                current_sample_ids = df.index.values
                feature_df = df
            
            # Ensure sample IDs are consistent
            if sample_ids is None:
                sample_ids = current_sample_ids
            elif not np.array_equal(sample_ids, current_sample_ids):
                raise IntegrationError(f"Inconsistent sample IDs in {omics_type}")
            
            # Scale features
            scaled_features = self._scale_features(feature_df, omics_type, fit=True)
            
            # Apply dimensionality reduction
            reduced_features = self._reduce_dimensionality(scaled_features, omics_type, fit=True)
            
            reduced_omics[omics_type] = reduced_features
        
        if not reduced_omics:
            raise IntegrationError("No reduced omics data available for integration")
        
        # Combine reduced features
        combined_features = []
        for omics_type, reduced_df in reduced_omics.items():
            # Add prefix to identify omics type
            reduced_df = reduced_df.add_prefix(f"{omics_type}_PC")
            combined_features.append(reduced_df)
        
        integrated_features = pd.concat(combined_features, axis=1)
        
        # Add clinical data if provided
        if clinical_data is not None and not clinical_data.empty:
            clinical_features = self._prepare_clinical_data(clinical_data, sample_ids)
            integrated_features = pd.concat([integrated_features, clinical_features], axis=1)
        
        # Create final integrated dataset
        integrated_data = pd.DataFrame({
            'sample_id': sample_ids
        })
        
        integrated_data = pd.concat([integrated_data, integrated_features], axis=1)
        
        result = {
            'integrated_data': integrated_data,
            'n_samples': len(sample_ids),
            'n_features': len(integrated_features.columns),
            'omics_types': list(reduced_omics.keys()),
            'method': 'intermediate_integration',
            'feature_names': list(integrated_features.columns),
            'reduction_info': self._get_reduction_info(),
            'scaling_info': self._get_scaling_info()
        }
        
        logger.info(f"Intermediate integration complete: {result['n_samples']} samples, {result['n_features']} features")
        
        return result
    
    def _scale_features(self, feature_df: pd.DataFrame, omics_type: str, 
                       fit: bool = True) -> pd.DataFrame:
        """Scale features for a specific omics type."""
        logger.debug(f"Scaling features for {omics_type}")
        
        if feature_df.empty:
            return feature_df
        
        # Remove any non-numeric columns
        numeric_df = feature_df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return numeric_df
        
        scaler_key = f"{omics_type}_scaler"
        
        if self.scaling_method == 'standard':
            if fit:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_df)
                self.fitted_scalers[scaler_key] = scaler
            else:
                scaler = self.fitted_scalers.get(scaler_key)
                if scaler is None:
                    logger.warning(f"No fitted scaler for {omics_type}")
                    return numeric_df
                scaled_data = scaler.transform(numeric_df)
        
        elif self.scaling_method == 'minmax':
            if fit:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(numeric_df)
                self.fitted_scalers[scaler_key] = scaler
            else:
                scaler = self.fitted_scalers.get(scaler_key)
                if scaler is None:
                    logger.warning(f"No fitted scaler for {omics_type}")
                    return numeric_df
                scaled_data = scaler.transform(numeric_df)
        
        else:
            logger.warning(f"Unknown scaling method: {self.scaling_method}")
            return numeric_df
        
        # Create scaled DataFrame
        scaled_df = pd.DataFrame(
            scaled_data,
            index=numeric_df.index,
            columns=numeric_df.columns
        )
        
        return scaled_df
    
    def _prepare_clinical_data(self, clinical_df: pd.DataFrame, 
                              sample_ids: np.ndarray) -> pd.DataFrame:
        """Prepare clinical data for integration."""
        logger.info("Preparing clinical data")
        
        # Ensure sample IDs are present
        if 'sample_id' not in clinical_df.columns:
            # Assume index contains sample IDs
            clinical_df = clinical_df.reset_index()
        
        # Filter to matching samples
        clinical_filtered = clinical_df[clinical_df['sample_id'].isin(sample_ids)].copy()
        
        # Sort to match sample order
        sample_id_series = pd.Series(sample_ids, name='sample_id')
        clinical_merged = pd.merge(sample_id_series, clinical_filtered, on='sample_id', how='left')
        
        # Separate numeric and categorical columns
        numeric_cols = clinical_merged.select_dtypes(include=[np.number]).columns
        categorical_cols = clinical_merged.select_dtypes(exclude=[np.number]).columns
        
        # Remove sample_id from processing
        numeric_cols = [col for col in numeric_cols if col != 'sample_id']
        categorical_cols = [col for col in categorical_cols if col != 'sample_id']
        
        # Scale numeric features
        if numeric_cols:
            numeric_features = clinical_merged[numeric_cols]
            
            scaler_key = "clinical_scaler"
            if scaler_key not in self.fitted_scalers:
                scaler = StandardScaler()
                scaled_numeric = scaler.fit_transform(numeric_features)
                self.fitted_scalers[scaler_key] = scaler
            else:
                scaler = self.fitted_scalers[scaler_key]
                scaled_numeric = scaler.transform(numeric_features)
            
            # Replace with scaled values
            clinical_merged[numeric_cols] = scaled_numeric
        
        # One-hot encode categorical features (if needed)
        # For now, keep categorical features as-is
        
        # Remove sample_id column, return only features
        feature_df = clinical_merged.drop(columns=['sample_id'])
        
        # Add prefix to identify clinical features
        feature_df = feature_df.add_prefix("clinical_")
        
        logger.info(f"Prepared clinical features: {feature_df.shape}")
        
        return feature_df
    
    def _reduce_dimensionality(self, feature_df: pd.DataFrame, omics_type: str, 
                              fit: bool = True, n_components: int = 50) -> pd.DataFrame:
        """Apply dimensionality reduction to features."""
        logger.debug(f"Reducing dimensionality for {omics_type}")
        
        if feature_df.empty or feature_df.shape[1] <= n_components:
            logger.info(f"Skipping dimensionality reduction for {omics_type}: already low-dimensional")
            return feature_df
        
        from sklearn.decomposition import PCA
        
        pca_key = f"{omics_type}_pca"
        
        if fit:
            pca = PCA(n_components=n_components, random_state=42)
            reduced_data = pca.fit_transform(feature_df)
            self.fitted_scalers[pca_key] = pca
            
            logger.info(f"PCA for {omics_type}: {feature_df.shape[1]} -> {pca.n_components_} components, "
                       f"explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        else:
            pca = self.fitted_scalers.get(pca_key)
            if pca is None:
                logger.warning(f"No fitted PCA for {omics_type}")
                return feature_df
            reduced_data = pca.transform(feature_df)
        
        # Create reduced DataFrame
        reduced_df = pd.DataFrame(
            reduced_data,
            index=feature_df.index,
            columns=[f"{omics_type}_PC{i+1}" for i in range(reduced_data.shape[1])]
        )
        
        return reduced_df
    
    def _calculate_integration_metrics(self, integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for integration."""
        logger.info("Calculating integration quality metrics")
        
        metrics = {
            'alignment_metrics': {},
            'integration_quality': {},
            'feature_metrics': {}
        }
        
        # Sample alignment metrics
        aligned_data = integration_results.get('aligned_data', {})
        if aligned_data:
            sample_counts = {omics: len(df) for omics, df in aligned_data.items()}
            common_samples = min(sample_counts.values()) if sample_counts else 0
            
            metrics['alignment_metrics'] = {
                'total_omics_types': len(aligned_data),
                'sample_counts_per_omics': sample_counts,
                'common_samples': common_samples,
                'alignment_percentage': (common_samples / max(sample_counts.values()) * 100) if sample_counts else 0
            }
        
        # Integration-specific metrics
        for method, result in integration_results.get('integration_methods', {}).items():
            if isinstance(result, dict) and 'status' not in result:
                # Calculate feature statistics
                integrated_data = result.get('integrated_data')
                if integrated_data is not None and not integrated_data.empty:
                    n_samples = result.get('n_samples', 0)
                    n_features = result.get('n_features', 0)
                    
                    # Feature density (non-zero features)
                    if n_samples > 0 and n_features > 0:
                        numeric_data = integrated_data.select_dtypes(include=[np.number])
                        non_zero_ratio = (numeric_data != 0).sum().sum() / (n_samples * n_features)
                        
                        metrics['integration_quality'][method] = {
                            'n_samples': n_samples,
                            'n_features': n_features,
                            'feature_density': non_zero_ratio,
                            'samples_per_feature': n_samples / n_features if n_features > 0 else 0
                        }
        
        return metrics
    
    def _get_scaling_info(self) -> Dict[str, Any]:
        """Get information about applied scaling."""
        return {
            'scaling_method': self.scaling_method,
            'fitted_scalers': list(self.fitted_scalers.keys()),
            'n_scalers': len(self.fitted_scalers)
        }
    
    def _get_reduction_info(self) -> Dict[str, Any]:
        """Get information about dimensionality reduction."""
        pca_scalers = {k: v for k, v in self.fitted_scalers.items() if 'pca' in k}
        
        reduction_info = {}
        for key, pca in pca_scalers.items():
            if hasattr(pca, 'explained_variance_ratio_'):
                reduction_info[key] = {
                    'n_components': pca.n_components_,
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'total_explained_variance': pca.explained_variance_ratio_.sum()
                }
        
        return reduction_info
    
    def get_integration_summary(self, integration_results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table of integration results."""
        summary_data = []
        
        for method, result in integration_results.get('integration_methods', {}).items():
            if isinstance(result, dict) and 'status' not in result:
                row = {
                    'integration_method': method,
                    'n_samples': result.get('n_samples', 0),
                    'n_features': result.get('n_features', 0),
                    'omics_types': ', '.join(result.get('omics_types', [])),
                    'status': 'success'
                }
                summary_data.append(row)
            else:
                row = {
                    'integration_method': method,
                    'n_samples': 0,
                    'n_features': 0,
                    'omics_types': '',
                    'status': result.get('status', 'failed') if isinstance(result, dict) else 'failed'
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)