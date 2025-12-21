"""
Module de normalisation des données omiques - Version simplifiée pour livraison rapide
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

class OmicsNormalizer:
    """Normalise les données omiques selon différentes méthodes - Version accélérée"""
    
    def __init__(self, method: str = 'log2_scale'):
        self.method = method
        self.logger = logging.getLogger('OmicsNormalizer')
    
    def normalize(self, data: pd.DataFrame, 
                  gene_lengths: Optional[pd.Series] = None) -> pd.DataFrame:
        """Normalise les données selon la méthode choisie"""
        self.logger.info(f"Normalisation avec méthode : {self.method}")
        
        if self.method == 'log2_scale':
            return self._log2_scaling_normalization(data)
        elif self.method == 'tmm':
            return self._tmm_normalization(data)
        elif self.method == 'tpm':
            if gene_lengths is None:
                self.logger.warning("Gene lengths required for TPM, using log2_scale instead")
                return self._log2_scaling_normalization(data)
            return self._tpm_transformation(data, gene_lengths)
        elif self.method == 'zscore':
            return self._zscore_normalization(data)
        else:
            raise ValueError(f"Méthode '{self.method}' non supportée")
    
    def _log2_scaling_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalisation log2 + scaling - Méthode simple et rapide"""
        self.logger.info("Normalisation log2 + scaling")
        
        # Séparer les colonnes numériques des identifiants
        id_cols = data.select_dtypes(exclude=[np.number]).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("Aucune colonne numérique trouvée")
            return data
        
        # Log2 transformation (x + 1 pour éviter log(0))
        normalized_data = data.copy()
        normalized_data[numeric_cols] = np.log2(normalized_data[numeric_cols] + 1)
        
        # Scaling (standardisation)
        for col in numeric_cols:
            mean_val = normalized_data[col].mean()
            std_val = normalized_data[col].std()
            if std_val > 0:
                normalized_data[col] = (normalized_data[col] - mean_val) / std_val
        
        self.logger.info("✅ Normalisation log2 + scaling terminée")
        return normalized_data
    
    def _tmm_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """TMM (Trimmed Mean of M-values) - Version simplifiée"""
        self.logger.info("Normalisation TMM simplifiée")
        
        try:
            # Sélectionner les colonnes numériques
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                self.logger.warning("Pas assez de colonnes pour TMM, utilisation log2_scale")
                return self._log2_scaling_normalization(data)
            
            # Calculer les facteurs de normalisation TMM simplifié
            expression_matrix = data[numeric_cols]
            
            # Log transformation
            log_expr = np.log2(expression_matrix + 1)
            
            # Calculer les moyennes géométriques par échantillon
            sample_means = log_expr.mean(axis=0)
            
            # Calculer les facteurs de normalisation
            # Ici, une version simplifiée - dans la vraie implémentation, 
            # on utiliserait une référence et trimmerait les outliers
            reference = sample_means.median()
            norm_factors = sample_means - reference
            
            # Appliquer les facteurs de normalisation
            normalized = log_expr.subtract(norm_factors, axis=1)
            
            # Remplacer dans le DataFrame original
            result = data.copy()
            result[numeric_cols] = normalized
            
            self.logger.info("✅ Normalisation TMM simplifiée terminée")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur TMM : {e}, fallback sur log2_scale")
            return self._log2_scaling_normalization(data)
    
    def _tpm_transformation(self, data: pd.DataFrame, 
                          gene_lengths: pd.Series) -> pd.DataFrame:
        """TPM (Transcripts Per Million) - Version simplifiée"""
        self.logger.info("Transformation TPM simplifiée")
        
        try:
            # S'assurer que les longueurs de gènes correspondent
            common_genes = data.columns.intersection(gene_lengths.index)
            if len(common_genes) == 0:
                self.logger.warning("Aucun gène commun trouvé, fallback sur log2_scale")
                return self._log2_scaling_normalization(data)
            
            # Filtrer les données et longueurs
            expr_data = data[common_genes]
            lengths = gene_lengths[common_genes]
            
            # Calculer RPK (Reads Per Kilobase)
            rpk = expr_data.divide(lengths / 1000, axis=1)
            
            # Calculer le facteur de normalisation (per million)
            scaling_factor = rpk.sum(axis=1) / 1_000_000
            
            # Calculer TPM
            tpm = rpk.divide(scaling_factor, axis=0)
            
            # Remplacer dans le DataFrame original
            result = data.copy()
            result[common_genes] = tpm
            
            self.logger.info("✅ Transformation TPM terminée")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur TPM : {e}, fallback sur log2_scale")
            return self._log2_scaling_normalization(data)
    
    def _zscore_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalisation Z-score (standardisation)"""
        self.logger.info("Normalisation Z-score")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("Aucune colonne numérique trouvée")
            return data
        
        result = data.copy()
        result[numeric_cols] = (result[numeric_cols] - result[numeric_cols].mean()) / result[numeric_cols].std()
        
        self.logger.info("✅ Normalisation Z-score terminée")
        return result
    
    def quantile_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalisation quantile - Pour données avec distributions très différentes"""
        self.logger.info("Normalisation quantile")
        
        try:
            from sklearn.preprocessing import QuantileTransformer
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return data
            
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            
            result = data.copy()
            result[numeric_cols] = qt.fit_transform(result[numeric_cols])
            
            self.logger.info("✅ Normalisation quantile terminée")
            return result
            
        except ImportError:
            self.logger.warning("QuantileTransformer non disponible, fallback sur log2_scale")
            return self._log2_scaling_normalization(data)

# Tests rapides
if __name__ == "__main__":
    # Créer des données de test
    np.random.seed(42)
    test_data = pd.DataFrame({
        'gene1': np.random.lognormal(8, 1.5, 10),
        'gene2': np.random.lognormal(7, 1.2, 10),
        'gene3': np.random.lognormal(6, 1.0, 10),
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    # Ajouter quelques zéros pour tester
    test_data.loc[2, 'gene1'] = 0
    test_data.loc[5, 'gene2'] = 0
    
    print("=== TEST NORMALISATION ===")
    print("Données originales :")
    print(test_data)
    print(f"\nStats originales:\n{test_data.describe()}")
    
    # Test log2_scale
    normalizer = OmicsNormalizer(method='log2_scale')
    normalized_data = normalizer.normalize(test_data.copy())
    
    print("\nAprès normalisation log2_scale :")
    print(normalized_data)
    print(f"\nStats normalisées:\n{normalized_data.describe()}")