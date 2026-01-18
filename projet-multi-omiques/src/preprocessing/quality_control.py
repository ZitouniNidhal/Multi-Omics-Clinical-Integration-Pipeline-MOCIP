"""Quality control for multi-omics data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging
from ..exceptions import QualityControlError

logger = logging.getLogger(__name__)


class QualityController:
    """Quality control for multi-omics datasets."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize quality controller.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enable_outlier_detection = self.config.get('enable_outlier_detection', True)
        self.outlier_method = self.config.get('outlier_method', 'isolation_forest')
        self.contamination = self.config.get('contamination', 0.05)
        self.qc_metrics = {}  # Store QC metrics
        
        logger.info(f"Initialized QualityController with method: {self.outlier_method}")
    
    def run_qc(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
              sample_groups: Optional[pd.Series] = None, **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive quality control.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            sample_groups: Sample grouping information
            **kwargs: Additional parameters
            
        Returns:
            QC results dictionary
        """
        logger.info("Running comprehensive quality control")
        
        qc_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_quality': 'PASS',
            'warnings': [],
            'failed_tests': [],
            'metrics': {}
        }
        
        if isinstance(data, dict):
            # Multi-omics QC
            for omics_type, df in data.items():
                logger.info(f"Running QC for {omics_type}")
                
                if df.empty:
                    logger.warning(f"Empty DataFrame for {omics_type}, skipping QC")
                    continue
                
                try:
                    omics_qc = self._run_single_qc(df, omics_type, sample_groups, **kwargs)
                    qc_results['metrics'][omics_type] = omics_qc
                    
                    # Update overall quality
                    if omics_qc['quality_status'] == 'FAIL':
                        qc_results['overall_quality'] = 'FAIL'
                        qc_results['failed_tests'].extend([f"{omics_type}_{test}" for test in omics_qc['failed_tests']])
                    
                    qc_results['warnings'].extend([f"{omics_type}: {warn}" for warn in omics_qc['warnings']])
                    
                except Exception as e:
                    logger.error(f"QC failed for {omics_type}: {e}")
                    qc_results['metrics'][omics_type] = {'quality_status': 'ERROR', 'error': str(e)}
                    qc_results['overall_quality'] = 'FAIL'
        
        else:
            # Single dataset QC
            qc_results['metrics']['single_dataset'] = self._run_single_qc(data, 'dataset', sample_groups, **kwargs)
            
            if qc_results['metrics']['single_dataset']['quality_status'] == 'FAIL':
                qc_results['overall_quality'] = 'FAIL'
                qc_results['failed_tests'].extend(qc_results['metrics']['single_dataset']['failed_tests'])
            
            qc_results['warnings'].extend(qc_results['metrics']['single_dataset']['warnings'])
        
        # Generate QC report
        qc_results['report'] = self._generate_qc_report(qc_results)
        
        logger.info(f"QC complete. Overall quality: {qc_results['overall_quality']}")
        
        return qc_results
    
    def _run_single_qc(self, df: pd.DataFrame, data_type: str, 
                      sample_groups: Optional[pd.Series] = None, **kwargs) -> Dict[str, Any]:
        """Run QC for single dataset."""
        logger.info(f"Running QC for {data_type} dataset")
        
        qc_result = {
            'data_type': data_type,
            'quality_status': 'PASS',
            'warnings': [],
            'failed_tests': [],
            'metrics': {}
        }
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(df)
        qc_result['metrics']['basic'] = basic_metrics
        
        # Missing data assessment
        missing_metrics = self._assess_missing_data(df)
        qc_result['metrics']['missing_data'] = missing_metrics
        
        if missing_metrics['missing_percentage'] > 20:
            qc_result['warnings'].append(f"High missing data: {missing_metrics['missing_percentage']:.1f}%")
        
        # Outlier detection
        if self.enable_outlier_detection:
            outlier_result = self._detect_outliers(df)
            qc_result['metrics']['outliers'] = outlier_result
            
            if outlier_result['outlier_percentage'] > self.contamination * 100:
                qc_result['warnings'].append(
                    f"High outlier rate: {outlier_result['outlier_percentage']:.1f}%"
                )
        
        # Distribution assessment
        distribution_metrics = self._assess_distributions(df)
        qc_result['metrics']['distributions'] = distribution_metrics
        
        # Correlation analysis
        correlation_metrics = self._analyze_correlations(df)
        qc_result['metrics']['correlations'] = correlation_metrics
        
        # Batch effects (if sample groups provided)
        if sample_groups is not None:
            batch_metrics = self._assess_batch_effects(df, sample_groups)
            qc_result['metrics']['batch_effects'] = batch_metrics
            
            if batch_metrics.get('significant_batch_effects', False):
                qc_result['warnings'].append("Potential batch effects detected")
        
        # Omics-specific QC
        if data_type == 'gene_expression':
            expression_metrics = self._assess_expression_quality(df)
            qc_result['metrics']['expression_quality'] = expression_metrics
            
            if expression_metrics.get('low_expression_genes_percentage', 0) > 50:
                qc_result['warnings'].append("High percentage of low-expression genes")
        
        # Determine pass/fail status
        qc_result = self._determine_qc_status(qc_result)
        
        return qc_result
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic dataset metrics."""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_columns': df.columns[df.columns.duplicated()].tolist()
        }
    
    def _assess_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess missing data patterns."""
        total_values = df.shape[0] * df.shape[1]
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / total_values) * 100 if total_values > 0 else 0
        
        # Per-column missing data
        col_missing = df.isnull().sum()
        col_missing_pct = (col_missing / len(df)) * 100
        
        # Rows with missing data
        rows_with_missing = df.isnull().any(axis=1).sum()
        rows_missing_pct = (rows_with_missing / len(df)) * 100
        
        # Missing data patterns
        missing_patterns = df.isnull().sum(axis=1).value_counts().head(10)
        
        return {
            'total_values': total_values,
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'columns_with_missing': int((col_missing > 0).sum()),
            'max_missing_per_column_pct': col_missing_pct.max() if len(col_missing_pct) > 0 else 0,
            'rows_with_missing': rows_with_missing,
            'rows_missing_percentage': rows_missing_pct,
            'complete_cases': len(df) - rows_with_missing,
            'missing_patterns': missing_patterns.to_dict()
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using various methods."""
        logger.info("Detecting outliers")
        
        # Use only numeric data
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'outliers': [], 'outlier_percentage': 0, 'method': 'none'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        outlier_results = {}
        
        if self.outlier_method == 'isolation_forest':
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            outlier_labels = iso_forest.fit_predict(scaled_data)
            outliers = np.where(outlier_labels == -1)[0]
            
            outlier_results = {
                'method': 'isolation_forest',
                'outliers': outliers.tolist(),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'n_estimators': 100,
                'contamination': self.contamination
            }
        
        elif self.outlier_method == 'iqr':
            # Interquartile Range method
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
            outliers = np.where(outlier_mask)[0]
            
            outlier_results = {
                'method': 'iqr',
                'outliers': outliers.tolist(),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'iqr_multiplier': 1.5
            }
        
        elif self.outlier_method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
            outlier_mask = (z_scores > 3).any(axis=1)
            outliers = np.where(outlier_mask)[0]
            
            outlier_results = {
                'method': 'zscore',
                'outliers': outliers.tolist(),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'zscore_threshold': 3.0
            }
        
        logger.info(f"Detected {len(outliers)} outliers ({outlier_results['outlier_percentage']:.1f}%)")
        
        return outlier_results
    
    def _assess_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data distributions."""
        logger.info("Assessing data distributions")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numeric data for distribution assessment'}
        
        distribution_metrics = {}
        
        for col in numeric_df.columns[:10]:  # Limit to first 10 columns for efficiency
            col_data = numeric_df[col].dropna()
            
            if len(col_data) < 10:
                continue
            
            # Basic statistics
            stats_dict = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data),
                'min': col_data.min(),
                'max': col_data.max()
            }
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
            if len(col_data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(col_data)
                    stats_dict['shapiro_wilk'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
                except:
                    stats_dict['shapiro_wilk'] = {'statistic': None, 'p_value': None}
            
            distribution_metrics[col] = stats_dict
        
        # Overall distribution summary
        all_data = numeric_df.values.flatten()
        all_data = all_data[~np.isnan(all_data)]
        
        if len(all_data) > 0:
            distribution_metrics['overall'] = {
                'global_mean': np.mean(all_data),
                'global_std': np.std(all_data),
                'global_skewness': stats.skew(all_data),
                'global_kurtosis': stats.kurtosis(all_data)
            }
        
        return distribution_metrics
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between features."""
        logger.info("Analyzing correlations")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return {'message': 'Insufficient numeric data for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.9:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Calculate eigenvalues for dimensionality assessment
        if numeric_df.shape[1] > 1 and numeric_df.shape[0] > numeric_df.shape[1]:
            try:
                pca = PCA()
                pca.fit(numeric_df.fillna(numeric_df.mean()))
                eigenvalues = pca.explained_variance_
                
                # Condition number
                condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf
                
                correlation_metrics = {
                    'high_correlation_pairs': high_corr_pairs,
                    'num_high_correlations': len(high_corr_pairs),
                    'max_correlation': corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                    'mean_correlation': corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                    'condition_number': condition_number,
                    'eigenvalue_ratio': eigenvalues[0] / eigenvalues[1] if len(eigenvalues) > 1 else np.inf
                }
            except:
                correlation_metrics = {
                    'high_correlation_pairs': high_corr_pairs,
                    'num_high_correlations': len(high_corr_pairs),
                    'max_correlation': corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                    'mean_correlation': corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                }
        else:
            correlation_metrics = {
                'high_correlation_pairs': high_corr_pairs,
                'num_high_correlations': len(high_corr_pairs),
                'max_correlation': corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                'mean_correlation': corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            }
        
        return correlation_metrics
    
    def _assess_batch_effects(self, df: pd.DataFrame, sample_groups: pd.Series) -> Dict[str, Any]:
        """Assess potential batch effects."""
        logger.info("Assessing batch effects")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numeric data for batch effect assessment'}
        
        # Align indices
        common_samples = numeric_df.index.intersection(sample_groups.index)
        if len(common_samples) == 0:
            return {'message': 'No common samples between data and groups'}
        
        numeric_df_aligned = numeric_df.loc[common_samples]
        groups_aligned = sample_groups.loc[common_samples]
        
        batch_metrics = {}
        
        # For each group, calculate mean and variance
        group_stats = {}
        for group in groups_aligned.unique():
            group_mask = groups_aligned == group
            group_data = numeric_df_aligned[group_mask]
            
            if len(group_data) > 1:
                group_stats[group] = {
                    'mean': group_data.mean(),
                    'std': group_data.std(),
                    'count': len(group_data)
                }
        
        # Calculate between-group to within-group variance ratio
        if len(group_stats) > 1:
            # Perform simple ANOVA-like analysis for first few features
            significant_features = []
            
            for feature in numeric_df_aligned.columns[:10]:  # Limit for efficiency
                feature_data = numeric_df_aligned[feature]
                
                # Group data
                groups_data = []
                for group, stats in group_stats.items():
                    if stats['count'] > 1:
                        group_values = feature_data[groups_aligned == group]
                        groups_data.append(group_values.dropna())
                
                if len(groups_data) > 1:
                    try:
                        # Kruskal-Wallis test (non-parametric)
                        h_stat, p_value = stats.kruskal(*groups_data)
                        
                        if p_value < 0.05:
                            significant_features.append({
                                'feature': feature,
                                'h_statistic': h_stat,
                                'p_value': p_value
                            })
                    except:
                        continue
            
            batch_metrics = {
                'num_groups': len(group_stats),
                'group_sizes': [stats['count'] for stats in group_stats.values()],
                'significant_batch_features': significant_features,
                'significant_batch_effects': len(significant_features),
                'significant_batch_effects_percentage': (len(significant_features) / min(10, len(numeric_df_aligned.columns))) * 100
            }
            
            # Flag potential batch effects
            if len(significant_features) > len(numeric_df_aligned.columns) * 0.1:
                batch_metrics['significant_batch_effects'] = True
            else:
                batch_metrics['significant_batch_effects'] = False
        
        return batch_metrics
    
    def _assess_expression_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess gene expression data quality."""
        logger.info("Assessing expression data quality")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numeric expression data'}
        
        # Calculate expression statistics
        gene_means = numeric_df.mean(axis=1)
        gene_stds = numeric_df.std(axis=1)
        sample_means = numeric_df.mean(axis=0)
        sample_stds = numeric_df.std(axis=0)
        
        # Detect low expression genes
        low_expression_threshold = gene_means.quantile(0.25)
        low_expression_genes = (gene_means < low_expression_threshold).sum()
        
        # Detect highly variable genes
        cv_genes = gene_stds / gene_means
        high_var_genes = (cv_genes > cv_genes.quantile(0.9)).sum()
        
        # Library size assessment
        library_sizes = numeric_df.sum(axis=0)
        
        # Detect samples with unusual library sizes
        lib_size_q1 = library_sizes.quantile(0.25)
        lib_size_q3 = library_sizes.quantile(0.75)
        lib_size_iqr = lib_size_q3 - lib_size_q1
        unusual_lib_sizes = ((library_sizes < lib_size_q1 - 1.5 * lib_size_iqr) | 
                           (library_sizes > lib_size_q3 + 1.5 * lib_size_iqr)).sum()
        
        expression_metrics = {
            'num_genes': len(numeric_df),
            'num_samples': len(numeric_df.columns),
            'mean_expression_per_gene': gene_means.mean(),
            'mean_expression_per_sample': sample_means.mean(),
            'low_expression_genes': int(low_expression_genes),
            'low_expression_genes_percentage': (low_expression_genes / len(numeric_df)) * 100,
            'highly_variable_genes': int(high_var_genes),
            'mean_cv_genes': cv_genes.mean(),
            'library_size_mean': library_sizes.mean(),
            'library_size_std': library_sizes.std(),
            'unusual_library_sizes': int(unusual_lib_sizes),
            'unusual_library_sizes_percentage': (unusual_lib_sizes / len(numeric_df.columns)) * 100
        }
        
        return expression_metrics
    
    def _determine_qc_status(self, qc_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall QC status based on metrics."""
        failed_tests = []
        
        # Basic thresholds
        if qc_result['metrics']['basic']['duplicate_rows'] > 0:
            failed_tests.append('duplicate_rows')
        
        if qc_result['metrics']['missing_data']['missing_percentage'] > 50:
            failed_tests.append('excessive_missing_data')
        
        if 'outliers' in qc_result['metrics']:
            if qc_result['metrics']['outliers']['outlier_percentage'] > 20:
                failed_tests.append('excessive_outliers')
        
        # Expression-specific thresholds
        if 'expression_quality' in qc_result['metrics']:
            expr_metrics = qc_result['metrics']['expression_quality']
            if expr_metrics.get('low_expression_genes_percentage', 0) > 70:
                failed_tests.append('excessive_low_expression_genes')
            if expr_metrics.get('unusual_library_sizes_percentage', 0) > 20:
                failed_tests.append('unusual_library_sizes')
        
        # Update QC status
        if failed_tests:
            qc_result['quality_status'] = 'FAIL'
            qc_result['failed_tests'] = failed_tests
        else:
            qc_result['quality_status'] = 'PASS'
        
        return qc_result
    
    def _generate_qc_report(self, qc_results: Dict[str, Any]) -> str:
        """Generate a detailed QC report."""
        report = []
        
        report.append("MULTI-OMICS QUALITY CONTROL REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {qc_results['timestamp']}")
        report.append(f"Overall Quality: {qc_results['overall_quality']}")
        report.append("")
        
        # Failed tests
        if qc_results['failed_tests']:
            report.append("FAILED TESTS:")
            for test in qc_results['failed_tests']:
                report.append(f"  - {test}")
            report.append("")
        
        # Warnings
        if qc_results['warnings']:
            report.append("WARNINGS:")
            for warning in qc_results['warnings']:
                report.append(f"  - {warning}")
            report.append("")
        
        # Detailed metrics for each data type
        for data_type, metrics in qc_results['metrics'].items():
            report.append(f"DATA TYPE: {data_type.upper()}")
            report.append("-" * 30)
            
            if 'quality_status' in metrics:
                report.append(f"Quality Status: {metrics['quality_status']}")
            
            # Basic metrics
            if 'basic' in metrics:
                basic = metrics['basic']
                report.append(f"Shape: {basic['shape']}")
                report.append(f"Memory Usage: {basic['memory_usage_mb']:.1f} MB")
                report.append(f"Duplicate Rows: {basic['duplicate_rows']}")
            
            # Missing data
            if 'missing_data' in metrics:
                missing = metrics['missing_data']
                report.append(f"Missing Data: {missing['missing_percentage']:.1f}%")
                report.append(f"Complete Cases: {missing['complete_cases']} ({missing['complete_cases_percentage']:.1f}%)")
            
            # Outliers
            if 'outliers' in metrics and isinstance(metrics['outliers'], dict):
                outliers = metrics['outliers']
                if 'outlier_percentage' in outliers:
                    report.append(f"Outliers: {outliers['outlier_percentage']:.1f}%")
            
            # Expression quality
            if 'expression_quality' in metrics:
                expr = metrics['expression_quality']
                report.append(f"Genes: {expr.get('num_genes', 'N/A')}")
                report.append(f"Samples: {expr.get('num_samples', 'N/A')}")
                if 'low_expression_genes_percentage' in expr:
                    report.append(f"Low Expression Genes: {expr['low_expression_genes_percentage']:.1f}%")
            
            report.append("")
        
        return "\n".join(report)
    
    def remove_outliers(self, df: pd.DataFrame, outlier_indices: List[int]) -> pd.DataFrame:
        """Remove detected outliers from dataset."""
        logger.info(f"Removing {len(outlier_indices)} outliers")
        
        # Create boolean mask for outliers
        outlier_mask = df.index.isin(outlier_indices)
        
        # Keep only non-outliers
        clean_df = df[~outlier_mask].copy()
        
        logger.info(f"Removed {outlier_mask.sum()} outliers. New shape: {clean_df.shape}")
        
        return clean_df
    
    def get_qc_metrics_summary(self, qc_results: Dict[str, Any]) -> pd.DataFrame:
        """Convert QC results to summary DataFrame."""
        summary_data = []
        
        for data_type, metrics in qc_results['metrics'].items():
            row = {
                'data_type': data_type,
                'quality_status': metrics.get('quality_status', 'UNKNOWN'),
                'shape': str(metrics.get('basic', {}).get('shape', 'N/A')),
                'missing_percentage': metrics.get('missing_data', {}).get('missing_percentage', np.nan),
                'outlier_percentage': metrics.get('outliers', {}).get('outlier_percentage', np.nan),
                'duplicate_rows': metrics.get('basic', {}).get('duplicate_rows', 0)
            }
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)