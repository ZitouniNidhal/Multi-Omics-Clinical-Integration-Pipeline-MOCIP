"""Multi-omics dimensionality reduction methods."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
from ..exceptions import IntegrationError

logger = logging.getLogger(__name__)


class MultiOmicsDimensionalityReduction:
    """Multi-omics dimensionality reduction methods."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize dimensionality reduction.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.method = self.config.get('method', 'mofa')
        self.n_components = self.config.get('n_components', 50)
        self.feature_selection = self.config.get('feature_selection', True)
        self.n_selected_features = self.config.get('n_selected_features', 1000)
        
        self.fitted_models = {}
        self.scalers = {}
        
        logger.info(f"Initialized MultiOmicsDimensionalityReduction with method: {self.method}")
    
    def fit_transform(self, omics_data: Dict[str, pd.DataFrame], 
                     target: Optional[pd.Series] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Fit dimensionality reduction model and transform data.
        
        Args:
            omics_data: Dictionary of omics DataFrames
            target: Target variable for supervised methods
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing reduced data and model information
        """
        logger.info(f"Fitting {self.method} dimensionality reduction")
        
        if self.method == 'mofa':
            return self._mofa_reduction(omics_data, target, fit=True, **kwargs)
        elif self.method == 'pca':
            return self._pca_reduction(omics_data, target, fit=True, **kwargs)
        elif self.method == 'pls':
            return self._pls_reduction(omics_data, target, fit=True, **kwargs)
        elif self.method == 'ica':
            return self._ica_reduction(omics_data, target, fit=True, **kwargs)
        elif self.method == 'nmf':
            return self._nmf_reduction(omics_data, target, fit=True, **kwargs)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.method}")
    
    def transform(self, omics_data: Dict[str, pd.DataFrame], 
                 **kwargs) -> Dict[str, Any]:
        """
        Transform data using fitted model.
        
        Args:
            omics_data: Dictionary of omics DataFrames
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing reduced data
        """
        logger.info(f"Transforming data with fitted {self.method} model")
        
        if self.method == 'mofa':
            return self._mofa_reduction(omics_data, fit=False, **kwargs)
        elif self.method == 'pca':
            return self._pca_reduction(omics_data, fit=False, **kwargs)
        elif self.method == 'pls':
            return self._pls_reduction(omics_data, fit=False, **kwargs)
        elif self.method == 'ica':
            return self._ica_reduction(omics_data, fit=False, **kwargs)
        elif self.method == 'nmf':
            return self._nmf_reduction(omics_data, fit=False, **kwargs)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.method}")
    
    def _mofa_reduction(self, omics_data: Dict[str, pd.DataFrame], 
                       target: Optional[pd.Series] = None,
                       fit: bool = True, **kwargs) -> Dict[str, Any]:
        """Multi-Omics Factor Analysis (MOFA) reduction."""
        logger.info("Applying MOFA dimensionality reduction")
        
        try:
            # Try to import mofapy2
            from mofapy2.run.entry_point import entry_point
            from mofapy2.simulate import simulate_mofa
        except ImportError:
            logger.warning("mofapy2 not available, using PCA as fallback")
            return self._pca_reduction(omics_data, target, fit=fit, **kwargs)
        
        try:
            # Prepare data for MOFA
            mofa_data = self._prepare_mofa_data(omics_data)
            
            if fit:
                # Create MOFA model
                ent = entry_point()
                
                # Set data
                ent.set_data_matrix(mofa_data['data'])
                ent.set_data_options(
                    scale_groups=False,
                    scale_views=True,
                    center_groups=False
                )
                
                # Set model options
                ent.set_model_options(
                    factors=self.n_components,
                    likelihooods=['gaussian'] * len(mofa_data['views']),
                    spikeslab_weights=True,
                    ard_weights=True
                )
                
                # Set training options
                ent.set_train_options(
                    convergence_mode='slow',
                    dropR2=0.001,
                    gpu_mode=False,
                    seed=42
                )
                
                # Build and train model
                ent.build()
                ent.run()
                
                # Store fitted model
                self.fitted_models['mofa'] = ent
                
                # Extract factors
                factors = ent.model.getExpectations()['Z']
                
            else:
                # Transform using fitted model
                ent = self.fitted_models.get('mofa')
                if ent is None:
                    raise IntegrationError("MOFA model not fitted")
                
                # Get factors for new data
                factors = ent.model.getExpectations()['Z']
            
            # Create reduced DataFrame
            sample_names = mofa_data['samples']
            factors_df = pd.DataFrame(
                factors,
                index=sample_names,
                columns=[f"MOFA_Factor_{i+1}" for i in range(factors.shape[1])]
            )
            
            # Calculate variance explained
            variance_explained = self._calculate_mofa_variance_explained(ent) if fit else {}
            
            result = {
                'reduced_data': factors_df,
                'method': 'mofa',
                'n_components': factors.shape[1],
                'sample_names': sample_names,
                'views': mofa_data['views'],
                'variance_explained': variance_explained,
                'model_info': {
                    'likelihoods': ['gaussian'] * len(mofa_data['views']),
                    'spikeslab_weights': True,
                    'ard_weights': True
                }
            }
            
            logger.info(f"MOFA reduction complete: {factors_df.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"MOFA reduction failed: {e}")
            # Fallback to PCA
            return self._pca_reduction(omics_data, target, fit=fit, **kwargs)
    
    def _pca_reduction(self, omics_data: Dict[str, pd.DataFrame], 
                      target: Optional[pd.Series] = None,
                      fit: bool = True, **kwargs) -> Dict[str, Any]:
        """Multi-view PCA reduction."""
        logger.info("Applying multi-view PCA reduction")
        
        # Prepare concatenated data
        concatenated_data, view_info = self._concatenate_omics_data(omics_data)
        
        if concatenated_data.empty:
            raise IntegrationError("No data available for PCA")
        
        # Scale data
        scaler_key = 'pca_scaler'
        if fit:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(concatenated_data)
            self.scalers[scaler_key] = scaler
        else:
            scaler = self.scalers.get(scaler_key)
            if scaler is None:
                raise IntegrationError("PCA scaler not fitted")
            scaled_data = scaler.transform(concatenated_data)
        
        # Apply PCA
        pca_key = 'pca_model'
        if fit:
            pca = PCA(n_components=self.n_components, random_state=42)
            components = pca.fit_transform(scaled_data)
            self.fitted_models[pca_key] = pca
            
            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}...")
            logger.info(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            
        else:
            pca = self.fitted_models.get(pca_key)
            if pca is None:
                raise IntegrationError("PCA model not fitted")
            components = pca.transform(scaled_data)
        
        # Create reduced DataFrame
        sample_names = concatenated_data.index
        components_df = pd.DataFrame(
            components,
            index=sample_names,
            columns=[f"PC{i+1}" for i in range(components.shape[1])]
        )
        
        # Calculate component loadings for each view
        loadings = self._calculate_pca_loadings(pca, view_info) if fit else {}
        
        result = {
            'reduced_data': components_df,
            'method': 'pca',
            'n_components': components.shape[1],
            'sample_names': sample_names,
            'explained_variance_ratio': pca.explained_variance_ratio_ if fit else None,
            'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_) if fit else None,
            'component_loadings': loadings,
            'view_info': view_info
        }
        
        logger.info(f"PCA reduction complete: {components_df.shape}")
        
        return result
    
    def _pls_reduction(self, omics_data: Dict[str, pd.DataFrame], 
                      target: pd.Series,
                      fit: bool = True, **kwargs) -> Dict[str, Any]:
        """Partial Least Squares (PLS) reduction."""
        logger.info("Applying PLS reduction")
        
        if target is None:
            logger.warning("No target provided for PLS, using PCA instead")
            return self._pca_reduction(omics_data, target, fit=fit, **kwargs)
        
        # Prepare concatenated data
        concatenated_data, view_info = self._concatenate_omics_data(omics_data)
        
        if concatenated_data.empty:
            raise IntegrationError("No data available for PLS")
        
        # Align target with data
        common_samples = concatenated_data.index.intersection(target.index)
        if len(common_samples) == 0:
            raise IntegrationError("No common samples between data and target")
        
        X_aligned = concatenated_data.loc[common_samples]
        y_aligned = target.loc[common_samples]
        
        # Scale data
        scaler_key = 'pls_scaler'
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_aligned)
            self.scalers[scaler_key] = scaler
        else:
            scaler = self.scalers.get(scaler_key)
            if scaler is None:
                raise IntegrationError("PLS scaler not fitted")
            X_scaled = scaler.transform(X_aligned)
        
        # Apply PLS
        pls_key = 'pls_model'
        if fit:
            pls = PLSRegression(n_components=self.n_components, scale=False)
            X_transformed = pls.fit_transform(X_scaled, y_aligned)[0]
            self.fitted_models[pls_key] = pls
            
            # Calculate R² score
            from sklearn.metrics import r2_score
            y_pred = pls.predict(X_scaled)
            r2 = r2_score(y_aligned, y_pred)
            
            logger.info(f"PLS R² score: {r2:.3f}")
            
        else:
            pls = self.fitted_models.get(pls_key)
            if pls is None:
                raise IntegrationError("PLS model not fitted")
            X_transformed = pls.transform(X_scaled)
        
        # Create reduced DataFrame
        components_df = pd.DataFrame(
            X_transformed,
            index=common_samples,
            columns=[f"PLS{i+1}" for i in range(X_transformed.shape[1])]
        )
        
        result = {
            'reduced_data': components_df,
            'method': 'pls',
            'n_components': X_transformed.shape[1],
            'sample_names': common_samples.tolist(),
            'target_variable': target.name if hasattr(target, 'name') else 'target',
            'r2_score': r2 if fit else None,
            'view_info': view_info
        }
        
        logger.info(f"PLS reduction complete: {components_df.shape}")
        
        return result
    
    def _ica_reduction(self, omics_data: Dict[str, pd.DataFrame], 
                      target: Optional[pd.Series] = None,
                      fit: bool = True, **kwargs) -> Dict[str, Any]:
        """Independent Component Analysis (ICA) reduction."""
        logger.info("Applying ICA reduction")
        
        # Prepare concatenated data
        concatenated_data, view_info = self._concatenate_omics_data(omics_data)
        
        if concatenated_data.empty:
            raise IntegrationError("No data available for ICA")
        
        # Scale data
        scaler_key = 'ica_scaler'
        if fit:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(concatenated_data)
            self.scalers[scaler_key] = scaler
        else:
            scaler = self.scalers.get(scaler_key)
            if scaler is None:
                raise IntegrationError("ICA scaler not fitted")
            scaled_data = scaler.transform(concatenated_data)
        
        # Apply ICA
        ica_key = 'ica_model'
        if fit:
            ica = FastICA(n_components=self.n_components, random_state=42)
            components = ica.fit_transform(scaled_data)
            self.fitted_models[ica_key] = ica
        else:
            ica = self.fitted_models.get(ica_key)
            if ica is None:
                raise IntegrationError("ICA model not fitted")
            components = ica.transform(scaled_data)
        
        # Create reduced DataFrame
        sample_names = concatenated_data.index
        components_df = pd.DataFrame(
            components,
            index=sample_names,
            columns=[f"IC{i+1}" for i in range(components.shape[1])]
        )
        
        result = {
            'reduced_data': components_df,
            'method': 'ica',
            'n_components': components.shape[1],
            'sample_names': sample_names,
            'view_info': view_info
        }
        
        logger.info(f"ICA reduction complete: {components_df.shape}")
        
        return result
    
    def _nmf_reduction(self, omics_data: Dict[str, pd.DataFrame], 
                      target: Optional[pd.Series] = None,
                      fit: bool = True, **kwargs) -> Dict[str, Any]:
        """Non-negative Matrix Factorization (NMF) reduction."""
        logger.info("Applying NMF reduction")
        
        # Prepare concatenated data
        concatenated_data, view_info = self._concatenate_omics_data(omics_data)
        
        if concatenated_data.empty:
            raise IntegrationError("No data available for NMF")
        
        # Ensure non-negative data
        if (concatenated_data < 0).any().any():
            logger.warning("Data contains negative values, shifting to non-negative")
            concatenated_data = concatenated_data - concatenated_data.min() + 1e-10
        
        # Apply NMF
        nmf_key = 'nmf_model'
        if fit:
            nmf = NMF(n_components=self.n_components, random_state=42, max_iter=1000)
            components = nmf.fit_transform(concatenated_data)
            self.fitted_models[nmf_key] = nmf
            
            reconstruction_error = nmf.reconstruction_err_
            logger.info(f"NMF reconstruction error: {reconstruction_error:.4f}")
            
        else:
            nmf = self.fitted_models.get(nmf_key)
            if nmf is None:
                raise IntegrationError("NMF model not fitted")
            components = nmf.transform(concatenated_data)
        
        # Create reduced DataFrame
        sample_names = concatenated_data.index
        components_df = pd.DataFrame(
            components,
            index=sample_names,
            columns=[f"NMF{i+1}" for i in range(components.shape[1])]
        )
        
        result = {
            'reduced_data': components_df,
            'method': 'nmf',
            'n_components': components.shape[1],
            'sample_names': sample_names,
            'reconstruction_error': reconstruction_error if fit else None,
            'view_info': view_info
        }
        
        logger.info(f"NMF reduction complete: {components_df.shape}")
        
        return result
    
    def _prepare_mofa_data(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare data for MOFA analysis."""
        # Convert to MOFA format
        views = []
        samples = None
        
        for omics_type, df in omics_data.items():
            # Extract numeric features
            if 'sample_id' in df.columns:
                feature_df = df.drop(columns=['sample_id'])
                current_samples = df['sample_id'].values
            else:
                feature_df = df
                current_samples = df.index.values
            
            # Ensure samples are consistent
            if samples is None:
                samples = current_samples
            elif not np.array_equal(samples, current_samples):
                # Reindex to match samples
                if 'sample_id' in df.columns:
                    feature_df = df.set_index('sample_id').reindex(samples).reset_index()
                else:
                    feature_df = df.reindex(samples)
            
            # Select only numeric columns
            numeric_df = feature_df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty:
                views.append(numeric_df.values.T)  # MOFA expects features x samples
            else:
                logger.warning(f"No numeric data for {omics_type}")
        
        return {
            'data': views,
            'views': list(omics_data.keys()),
            'samples': samples
        }
    
    def _concatenate_omics_data(self, omics_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Concatenate omics data horizontally."""
        concatenated_dfs = []
        view_info = {}
        start_idx = 0
        
        for omics_type, df in omics_data.items():
            # Extract numeric features
            if 'sample_id' in df.columns:
                feature_df = df.drop(columns=['sample_id'])
            else:
                feature_df = df
            
            # Select only numeric columns
            numeric_df = feature_df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty:
                # Add prefix to avoid column conflicts
                numeric_df = numeric_df.add_prefix(f"{omics_type}_")
                
                # Store view information
                end_idx = start_idx + len(numeric_df.columns)
                view_info[omics_type] = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'n_features': len(numeric_df.columns)
                }
                start_idx = end_idx
                
                concatenated_dfs.append(numeric_df)
        
        if concatenated_dfs:
            concatenated_data = pd.concat(concatenated_dfs, axis=1)
        else:
            concatenated_data = pd.DataFrame()
        
        return concatenated_data, view_info
    
    def _calculate_pca_loadings(self, pca: PCA, view_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate PCA loadings for each view (omics type)."""
        loadings = {}
        
        components = pca.components_  # shape: (n_components, n_features)
        
        for view_name, info in view_info.items():
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            
            # Extract loadings for this view
            view_loadings = components[:, start_idx:end_idx]
            loadings[view_name] = view_loadings
        
        return loadings
    
    def _calculate_mofa_variance_explained(self, ent) -> Dict[str, Any]:
        """Calculate variance explained by MOFA factors."""
        try:
            # Get variance explained per factor
            variance_explained = ent.model.calculate_variance_explained()
            
            # Organize by view and factor
            result = {
                'per_factor': {},
                'per_view': {},
                'total': {}
            }
            
            # This is a simplified implementation
            # Real MOFA variance calculation would be more complex
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to calculate MOFA variance explained: {e}")
            return {}
    
    def apply_feature_selection(self, omics_data: Dict[str, pd.DataFrame], 
                               target: pd.Series, method: str = 'mutual_info') -> Dict[str, pd.DataFrame]:
        """
        Apply feature selection to each omics type.
        
        Args:
            omics_data: Dictionary of omics DataFrames
            target: Target variable
            method: Feature selection method
            
        Returns:
            Dictionary of DataFrames with selected features
        """
        logger.info(f"Applying feature selection using {method}")
        
        selected_data = {}
        
        for omics_type, df in omics_data.items():
            logger.info(f"Selecting features for {omics_type}")
            
            # Extract features and align with target
            if 'sample_id' in df.columns:
                feature_df = df.set_index('sample_id')
            else:
                feature_df = df
            
            # Align with target
            common_samples = feature_df.index.intersection(target.index)
            if len(common_samples) == 0:
                logger.warning(f"No common samples for {omics_type}")
                selected_data[omics_type] = df
                continue
            
            X = feature_df.loc[common_samples]
            y = target.loc[common_samples]
            
            # Remove non-numeric columns
            X_numeric = X.select_dtypes(include=[np.number])
            
            if X_numeric.empty:
                logger.warning(f"No numeric features for {omics_type}")
                selected_data[omics_type] = df
                continue
            
            # Apply feature selection
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_selected_features, X_numeric.shape[1]))
            elif method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=min(self.n_selected_features, X_numeric.shape[1]))
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                selected_data[omics_type] = df
                continue
            
            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()]
            
            # Create DataFrame with selected features
            if 'sample_id' in df.columns:
                selected_df = df[['sample_id'] + list(selected_features)]
            else:
                selected_df = df[selected_features]
            
            selected_data[omics_type] = selected_df
            
            logger.info(f"Selected {len(selected_features)}/{X_numeric.shape[1]} features for {omics_type}")
        
        return selected_data
    
    def get_component_interpretation(self, reduction_results: Dict[str, Any], 
                                   n_top_features: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get interpretation of components (top contributing features).
        
        Args:
            reduction_results: Results from dimensionality reduction
            n_top_features: Number of top features to show
            
        Returns:
            Dictionary with component interpretations
        """
        method = reduction_results.get('method')
        
        if method == 'pca':
            return self._interpret_pca_components(reduction_results, n_top_features)
        elif method == 'mofa':
            return self._interpret_mofa_components(reduction_results, n_top_features)
        else:
            logger.warning(f"Component interpretation not implemented for {method}")
            return {}
    
    def _interpret_pca_components(self, pca_results: Dict[str, Any], 
                                 n_top_features: int) -> Dict[str, pd.DataFrame]:
        """Interpret PCA components."""
        component_loadings = pca_results.get('component_loadings', {})
        
        interpretations = {}
        
        for view_name, loadings in component_loadings.items():
            # loadings shape: (n_components, n_features_in_view)
            n_components, n_features = loadings.shape
            
            # Get feature names for this view
            view_info = pca_results.get('view_info', {})
            if view_name in view_info:
                # In practice, you'd store the actual feature names
                feature_names = [f"{view_name}_feature_{i}" for i in range(n_features)]
            else:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            
            # Find top features for each component
            component_interpretations = []
            
            for comp_idx in range(min(5, n_components)):  # Limit to first 5 components
                component_loading = loadings[comp_idx]
                
                # Get absolute loadings and sort
                abs_loadings = np.abs(component_loading)
                top_indices = np.argsort(abs_loadings)[::-1][:n_top_features]
                
                top_features = []
                for idx in top_indices:
                    top_features.append({
                        'component': f"PC{comp_idx + 1}",
                        'feature': feature_names[idx],
                        'loading': component_loading[idx],
                        'abs_loading': abs_loadings[idx]
                    })
                
                component_interpretations.extend(top_features)
            
            interpretations[view_name] = pd.DataFrame(component_interpretations)
        
        return interpretations