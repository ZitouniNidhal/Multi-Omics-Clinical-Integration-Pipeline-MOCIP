"""
Module de gestion des valeurs manquantes - Version simplifiée pour livraison rapide
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Union, List, Optional, Any, Dict
import logging

class MissingValueHandler:
    """Gère l'imputation des valeurs manquantes - Version accélérée"""
    
    def __init__(self, strategy: str = 'knn', k: int = 3):
        self.strategy = strategy
        self.k = k
        self.logger = logging.getLogger('MissingValueHandler')
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applique l'imputation selon la stratégie choisie"""
        self.logger.info(f"Imputation des valeurs manquantes avec stratégie : {self.strategy}")
        
        # Vérifier s'il y a des valeurs manquantes
        missing_before = data.isnull().sum().sum()
        if missing_before == 0:
            self.logger.info("Aucune valeur manquante détectée")
            return data
        
        self.logger.info(f"Valeurs manquantes avant imputation : {missing_before}")
        
        if self.strategy == 'knn':
            return self._knn_imputation(data)
        elif self.strategy == 'median':
            return self._median_imputation(data)
        elif self.strategy == 'mean':
            return self._mean_imputation(data)
        else:
            raise ValueError(f"Stratégie '{self.strategy}' non supportée")
    
    def _knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Imputation par KNN - Version simplifiée"""
        try:
            # Séparer les colonnes numériques et catégorielles
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns
            
            # Imputation pour les colonnes numériques
            if len(numeric_cols) > 0:
                numeric_data = data[numeric_cols].copy()
                imputer = KNNImputer(n_neighbors=min(self.k, len(numeric_data)))
                numeric_imputed = imputer.fit_transform(numeric_data)
                data[numeric_cols] = numeric_imputed
            
            # Imputation pour les colonnes catégorielles (mode)
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                    data[col].fillna(mode_value, inplace=True)
            
            missing_after = data.isnull().sum().sum()
            self.logger.info(f"✅ Imputation KNN terminée. Valeurs manquantes après : {missing_after}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'imputation KNN : {e}")
            # Fallback sur médiane
            return self._median_imputation(data)
    
    def _median_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Imputation par la médiane - Fallback simple"""
        self.logger.info("Imputation par médiane (fallback)")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Médiane pour numériques
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
        
        # Mode pour catégorielles
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col].fillna(mode_val, inplace=True)
        
        missing_after = data.isnull().sum().sum()
        self.logger.info(f"✅ Imputation par médiane terminée. Valeurs manquantes après : {missing_after}")
        
        return data
    
    def _mean_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Imputation par la moyenne"""
        self.logger.info("Imputation par moyenne")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Moyenne pour numériques
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                mean_val = data[col].mean()
                data[col].fillna(mean_val, inplace=True)
        
        # Mode pour catégorielles
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col].fillna(mode_val, inplace=True)
        
        missing_after = data.isnull().sum().sum()
        self.logger.info(f"✅ Imputation par moyenne terminée. Valeurs manquantes après : {missing_after}")
        
        return data
    
    def filter_low_quality_features(self, data: pd.DataFrame, 
                                  threshold: float = 0.5) -> pd.DataFrame:
        """Filtre les features avec trop de valeurs manquantes"""
        self.logger.info(f"Filtrage des features avec >{threshold*100}% valeurs manquantes")
        
        # Calculer le pourcentage de valeurs manquantes par colonne
        missing_percentages = data.isnull().sum() / len(data)
        
        # Identifier les colonnes à garder
        cols_to_keep = missing_percentages[missing_percentages <= threshold].index
        
        filtered_data = data[cols_to_keep].copy()
        
        removed_cols = len(data.columns) - len(cols_to_keep)
        self.logger.info(f"Features retirées : {removed_cols}, Features conservées : {len(cols_to_keep)}")
        
        return filtered_data
    
    def get_missing_value_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un rapport sur les valeurs manquantes"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        report = {
            'total_missing_values': missing_counts.sum(),
            'total_missing_percentage': (missing_counts.sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
        
        return report

# Tests rapides
if __name__ == "__main__":
    # Créer des données de test
    test_data = pd.DataFrame({
        'gene1': [1, 2, np.nan, 4, 5],
        'gene2': [2, np.nan, 4, 5, 6],
        'gene3': [1, 2, 3, 4, 5],
        'category': ['A', 'B', np.nan, 'A', 'B']
    })
    
    print("=== TEST MISSING VALUE HANDLER ===")
    print("Données originales :")
    print(test_data)
    print(f"Valeurs manquantes : {test_data.isnull().sum().sum()}")
    
    # Test KNN
    handler = MissingValueHandler(strategy='knn', k=2)
    imputed_data = handler.fit_transform(test_data.copy())
    
    print("\nAprès imputation KNN :")
    print(imputed_data)
    print(f"Valeurs manquantes après : {imputed_data.isnull().sum().sum()}")
    
    # Rapport
    report = handler.get_missing_value_report(test_data)
    print(f"\nRapport : {report}")