"""Machine Learning ready exporter for multi-omics data."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import joblib
import logging
from ..exceptions import ExportError

logger = logging.getLogger(__name__)


class MLExporter:
    """Export multi-omics data in ML-ready formats."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ML exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        self.stratify_by = self.config.get('stratify_by', None)
        self.scaling_method = self.config.get('scaling_method', 'standard')
        self.feature_selection_method = self.config.get('feature_selection', {}).get('method', 'mutual_info')
        self.n_features = self.config.get('feature_selection', {}).get('k_best', 1000)
        
        self.fitted_scalers = {}
        self.fitted_selectors = {}
        
        logger.info(f"Initialized MLExporter with test_size: {self.test_size}")
    
    def prepare_ml_data(self, integrated_data: Dict[str, Any], 
                       target_variable: str = 'target',
                       **kwargs) -> Dict[str, Any]:
        """
        Prepare ML-ready data from integrated multi-omics data.
        
        Args:
            integrated_data: Integrated multi-omics data
            target_variable: Name of target variable column
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing ML-ready data and metadata
        """
        logger.info("Preparing ML-ready data")
        
        try:
            # Extract features and target
            feature_data, target_data = self._extract_features_and_target(integrated_data, target_variable)
            
            if feature_data.empty:
                raise ExportError("No feature data available")
            
            if target_data.empty:
                raise ExportError("No target data available")
            
            # Align data
            aligned_features, aligned_target = self._align_features_and_target(feature_data, target_data)
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(aligned_features, aligned_target)
            
            # Scale features
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
            
            # Apply feature selection
            X_train_selected, X_test_selected, selected_features = self._select_features(
                X_train_scaled, X_test_scaled, y_train
            )
            
            # Create ML-ready dataset
            ml_data = {
                'X_train': X_train_selected,
                'X_test': X_test_selected,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': selected_features,
                'target_name': target_variable,
                'n_samples_train': len(X_train_selected),
                'n_samples_test': len(X_test_selected),
                'n_features': X_train_selected.shape[1],
                'scaling_method': self.scaling_method,
                'feature_selection_method': self.feature_selection_method,
                'preparation_timestamp': datetime.now().isoformat()
            }
            
            # Add metadata
            ml_data['metadata'] = self._create_ml_metadata(ml_data, integrated_data)
            
            logger.info(f"ML data preparation complete: {ml_data['n_samples_train']} train, "
                       f"{ml_data['n_samples_test']} test samples, {ml_data['n_features']} features")
            
            return ml_data
            
        except Exception as e:
            logger.error(f"ML data preparation failed: {e}")
            raise ExportError(f"ML data preparation failed: {e}")
    
    def save_ml_dataset(self, ml_data: Dict[str, Any], output_dir: str, 
                       save_models: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Save ML dataset to disk.
        
        Args:
            ml_data: ML-ready data dictionary
            output_dir: Output directory path
            save_models: Whether to save fitted models
            **kwargs: Additional parameters
            
        Returns:
            Save results
        """
        logger.info(f"Saving ML dataset to: {output_dir}")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Save training data
            train_data = {
                'X': ml_data['X_train'],
                'y': ml_data['y_train'],
                'feature_names': ml_data['feature_names']
            }
            
            train_file = output_path / "train_data.joblib"
            joblib.dump(train_data, train_file)
            saved_files['train_data'] = str(train_file)
            
            # Save test data
            test_data = {
                'X': ml_data['X_test'],
                'y': ml_data['y_test'],
                'feature_names': ml_data['feature_names']
            }
            
            test_file = output_path / "test_data.joblib"
            joblib.dump(test_data, test_file)
            saved_files['test_data'] = str(test_file)
            
            # Save fitted models if requested
            if save_models:
                if self.fitted_scalers:
                    scalers_file = output_path / "scalers.joblib"
                    joblib.dump(self.fitted_scalers, scalers_file)
                    saved_files['scalers'] = str(scalers_file)
                
                if self.fitted_selectors:
                    selectors_file = output_path / "feature_selectors.joblib"
                    joblib.dump(self.fitted_selectors, selectors_file)
                    saved_files['feature_selectors'] = str(selectors_file)
            
            # Save metadata
            metadata = ml_data.get('metadata', {})
            metadata_file = output_path / "metadata.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            saved_files['metadata'] = str(metadata_file)
            
            # Save summary report
            report = self._create_ml_summary_report(ml_data)
            report_file = output_path / "ml_dataset_summary.txt"
            
            with open(report_file, 'w') as f:
                f.write(report)
            saved_files['summary_report'] = str(report_file)
            
            result = {
                'status': 'success',
                'output_directory': str(output_path),
                'saved_files': saved_files,
                'n_train_samples': ml_data['n_samples_train'],
                'n_test_samples': ml_data['n_samples_test'],
                'n_features': ml_data['n_features'],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"ML dataset saved successfully: {len(saved_files)} files")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to save ML dataset: {e}")
            raise ExportError(f"Failed to save ML dataset: {e}")
    
    def _extract_features_and_target(self, integrated_data: Dict[str, Any], 
                                   target_variable: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target from integrated data."""
        # Get the main integrated dataset
        main_data = integrated_data.get('integrated_data')
        
        if main_data is None or main_data.empty:
            # Try to get from individual omics
            individual_omics = integrated_data.get('individual_omics', {})
            if individual_omics:
                # Concatenate individual omics data
                feature_dfs = []
                for omics_type, omics_data in individual_omics.items():
                    if isinstance(omics_data, dict) and 'features' in omics_data:
                        feature_dfs.append(omics_data['features'])
                    elif isinstance(omics_data, pd.DataFrame):
                        feature_dfs.append(omics_data)
                
                if feature_dfs:
                    main_data = pd.concat(feature_dfs, axis=1)
                else:
                    raise ExportError("No feature data found in integrated data")
            else:
                raise ExportError("No integrated data available")
        
        # Separate features and target
        if target_variable in main_data.columns:
            feature_data = main_data.drop(columns=[target_variable])
            target_data = main_data[target_variable]
        else:
            # Assume all columns are features, target must be provided separately
            feature_data = main_data
            target_data = pd.Series()  # Empty target
        
        logger.info(f"Extracted features: {feature_data.shape}, target: {len(target_data)} samples")
        
        return feature_data, target_data
    
    def _align_features_and_target(self, features: pd.DataFrame, 
                                 target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target data."""
        if target.empty:
            return features, target
        
        # Find common indices
        common_indices = features.index.intersection(target.index)
        
        if len(common_indices) == 0:
            raise ExportError("No common samples between features and target")
        
        # Align data
        aligned_features = features.loc[common_indices]
        aligned_target = target.loc[common_indices]
        
        logger.info(f"Aligned data: {len(common_indices)} common samples")
        
        return aligned_features, aligned_target
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        if y.empty:
            # Unsupervised case - split features only
            X_train, X_test = train_test_split(
                X, test_size=self.test_size, random_state=self.random_state
            )
            y_train = pd.Series()
            y_test = pd.Series()
        else:
            # Supervised case - stratified split if possible
            stratify = None
            if self.stratify_by and self.stratify_by in y.values:
                stratify = y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state,
                stratify=stratify
            )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using specified method."""
        if self.scaling_method == 'none':
            return X_train, X_test
        
        logger.info(f"Scaling features using {self.scaling_method}")
        
        # Choose scaler
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {self.scaling_method}, using standard")
            scaler = StandardScaler()
        
        # Fit on training data and transform both
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        
        # Store fitted scaler
        self.fitted_scalers['feature_scaler'] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def _select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Apply feature selection."""
        if self.feature_selection_method == 'none' or y_train.empty:
            return X_train, X_test, list(X_train.columns)
        
        logger.info(f"Selecting features using {self.feature_selection_method}")
        
        n_features = min(self.n_features, X_train.shape[1])
        
        # Choose feature selector
        if self.feature_selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        elif self.feature_selection_method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif self.feature_selection_method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            selector = RFE(estimator, n_features_to_select=n_features)
        else:
            logger.warning(f"Unknown feature selection method: {self.feature_selection_method}")
            return X_train, X_test, list(X_train.columns)
        
        # Fit selector and transform data
        if isinstance(selector, RFE):
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = [X_train.columns[i] for i in selector.get_support(indices=True)]
        else:
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()].tolist()
        
        # Create DataFrames with selected features
        X_train_selected_df = pd.DataFrame(
            X_train_selected,
            index=X_train.index,
            columns=selected_features
        )
        
        X_test_selected_df = pd.DataFrame(
            X_test_selected,
            index=X_test.index,
            columns=selected_features
        )
        
        # Store fitted selector
        self.fitted_selectors['feature_selector'] = selector
        
        logger.info(f"Selected {len(selected_features)} features from {X_train.shape[1]} original features")
        
        return X_train_selected_df, X_test_selected_df, selected_features
    
    def _create_ml_metadata(self, ml_data: Dict[str, Any], 
                          original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for ML dataset."""
        metadata = {
            'dataset_info': {
                'n_train_samples': ml_data['n_samples_train'],
                'n_test_samples': ml_data['n_samples_test'],
                'n_features': ml_data['n_features'],
                'feature_names': ml_data['feature_names'],
                'target_name': ml_data['target_name'],
                'scaling_method': ml_data['scaling_method'],
                'feature_selection_method': ml_data['feature_selection_method']
            },
            'preprocessing_info': {
                'test_size': self.test_size,
                'random_state': self.random_state,
                'stratify_by': self.stratify_by,
                'n_selected_features': self.n_features
            },
            'original_data_info': {
                'integration_method': original_data.get('integration_methods', {}).keys() if isinstance(original_data, dict) else 'unknown',
                'n_omics_types': len(original_data.get('aligned_data', {})) if isinstance(original_data, dict) else 0
            },
            'quality_metrics': self._calculate_ml_quality_metrics(ml_data),
            'creation_timestamp': datetime.now().isoformat()
        }
        
        return metadata
    
    def _calculate_ml_quality_metrics(self, ml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for ML dataset."""
        metrics = {}
        
        # Class balance (if classification)
        if not ml_data['y_train'].empty:
            train_class_counts = ml_data['y_train'].value_counts()
            test_class_counts = ml_data['y_test'].value_counts()
            
            metrics['class_balance'] = {
                'train': train_class_counts.to_dict(),
                'test': test_class_counts.to_dict(),
                'train_balance_ratio': train_class_counts.min() / train_class_counts.max() if train_class_counts.max() > 0 else 0,
                'test_balance_ratio': test_class_counts.min() / test_class_counts.max() if test_class_counts.max() > 0 else 0
            }
        
        # Feature statistics
        X_train = ml_data['X_train']
        
        metrics['feature_stats'] = {
            'mean_features': X_train.mean().mean(),
            'std_features': X_train.std().mean(),
            'min_feature_value': X_train.min().min(),
            'max_feature_value': X_train.max().max(),
            'feature_sparsity': (X_train == 0).sum().sum() / (X_train.shape[0] * X_train.shape[1])
        }
        
        # Sample statistics
        metrics['sample_stats'] = {
            'train_test_ratio': ml_data['n_samples_test'] / ml_data['n_samples_train'] if ml_data['n_samples_train'] > 0 else 0,
            'features_per_sample': ml_data['n_features'],
            'samples_per_feature': ml_data['n_samples_train'] / ml_data['n_features'] if ml_data['n_features'] > 0 else 0
        }
        
        return metrics
    
    def _create_ml_summary_report(self, ml_data: Dict[str, Any]) -> str:
        """Create summary report for ML dataset."""
        report = []
        
        report.append("MACHINE LEARNING DATASET SUMMARY")
        report.append("=" * 50)
        report.append(f"Creation Timestamp: {ml_data.get('preparation_timestamp', 'Unknown')}")
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append(f"  Training samples: {ml_data['n_samples_train']}")
        report.append(f"  Test samples: {ml_data['n_samples_test']}")
        report.append(f"  Features: {ml_data['n_features']}")
        report.append(f"  Target variable: {ml_data['target_name']}")
        report.append("")
        
        # Preprocessing info
        report.append("PREPROCESSING:")
        report.append(f"  Test size: {self.test_size}")
        report.append(f"  Random state: {self.random_state}")
        report.append(f"  Scaling method: {ml_data['scaling_method']}")
        report.append(f"  Feature selection: {ml_data['feature_selection_method']}")
        report.append(f"  Selected features: {ml_data['n_features']}")
        report.append("")
        
        # Quality metrics
        metadata = ml_data.get('metadata', {})
        quality_metrics = metadata.get('quality_metrics', {})
        
        if quality_metrics:
            report.append("QUALITY METRICS:")
            
            # Class balance
            if 'class_balance' in quality_metrics:
                cb = quality_metrics['class_balance']
                report.append(f"  Train class balance ratio: {cb['train_balance_ratio']:.3f}")
                report.append(f"  Test class balance ratio: {cb['test_balance_ratio']:.3f}")
            
            # Feature stats
            if 'feature_stats' in quality_metrics:
                fs = quality_metrics['feature_stats']
                report.append(f"  Feature sparsity: {fs['feature_sparsity']:.3f}")
                report.append(f"  Feature value range: [{fs['min_feature_value']:.2f}, {fs['max_feature_value']:.2f}]")
            
            report.append("")
        
        # Sample feature names
        report.append("SELECTED FEATURES (first 20):")
        feature_names = ml_data['feature_names'][:20]
        for i, feature in enumerate(feature_names, 1):
            report.append(f"  {i:2d}. {feature}")
        
        if len(ml_data['feature_names']) > 20:
            report.append(f"  ... and {len(ml_data['feature_names']) - 20} more features")
        
        return "\n".join(report)
    
    def quick_model_evaluation(self, ml_data: Dict[str, Any], 
                             model_type: str = 'random_forest', **kwargs) -> Dict[str, Any]:
        """
        Quick model evaluation on the prepared ML data.
        
        Args:
            ml_data: ML-ready data
            model_type: Type of model to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Evaluation results
        """
        logger.info(f"Running quick model evaluation with {model_type}")
        
        try:
            if ml_data['y_train'].empty:
                return {'error': 'No target variable available for evaluation'}
            
            # Choose model
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            elif model_type == 'svm':
                from sklearn.svm import SVC
                model = SVC(random_state=self.random_state)
            else:
                return {'error': f'Unknown model type: {model_type}'}
            
            # Train model
            model.fit(ml_data['X_train'], ml_data['y_train'])
            
            # Make predictions
            y_pred_train = model.predict(ml_data['X_train'])
            y_pred_test = model.predict(ml_data['X_test'])
            
            # Calculate metrics
            train_accuracy = model.score(ml_data['X_train'], ml_data['y_train'])
            test_accuracy = model.score(ml_data['X_test'], ml_data['y_test'])
            
            # Classification report
            test_report = classification_report(ml_data['y_test'], y_pred_test, output_dict=True)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.Series(
                    model.feature_importances_,
                    index=ml_data['feature_names']
                ).sort_values(ascending=False).head(20).to_dict()
            elif hasattr(model, 'coef_'):
                feature_importance = pd.Series(
                    np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_),
                    index=ml_data['feature_names']
                ).sort_values(ascending=False).head(20).to_dict()
            
            evaluation_results = {
                'model_type': model_type,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'classification_report': test_report,
                'confusion_matrix': confusion_matrix(ml_data['y_test'], y_pred_test).tolist(),
                'feature_importance': feature_importance,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Model evaluation complete - Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}