"""
Module de fusion des données multi-modalités - Version simplifiée pour livraison rapide
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

class MultiOmicsFusion:
    """Fusionne les données de différentes modalités omiques - Version accélérée"""
    
    def __init__(self, fusion_method: str = 'horizontal'):
        self.fusion_method = fusion_method
        self.logger = logging.getLogger('MultiOmicsFusion')
    
    def horizontal_fusion(self, datasets: Dict[str, pd.DataFrame], 
                         sample_key: str = 'patient_id') -> pd.DataFrame:
        """
        Fusion horizontale des datasets (concaténation des colonnes)
        
        Args:
            datasets: Dictionnaire des datasets à fusionner
            sample_key: Clé pour l'identifiant d'échantillon
        
        Returns:
            DataFrame fusionné
        """
        self.logger.info("Fusion horizontale des datasets")
        
        if len(datasets) < 2:
            self.logger.warning("Moins de 2 datasets à fusionner")
            return list(datasets.values())[0] if datasets else pd.DataFrame()
        
        # Vérifier que tous les datasets ont la même clé d'échantillon
        for name, df in datasets.items():
            if sample_key not in df.columns:
                self.logger.error(f"Clé {sample_key} manquante dans {name}")
                return pd.DataFrame()
        
        # Commencer avec le premier dataset
        dataset_names = list(datasets.keys())
        fused_data = datasets[dataset_names[0]].copy()
        
        self.logger.info(f"Dataset de base : {dataset_names[0]} ({fused_data.shape})")
        
        # Fusionner avec les autres datasets
        for i, name in enumerate(dataset_names[1:], 1):
            df = datasets[name].copy()
            
            # S'assurer que la clé d'échantillon est de même type
            fused_data[sample_key] = fused_data[sample_key].astype(str)
            df[sample_key] = df[sample_key].astype(str)
            
            # Fusionner sur la clé d'échantillon
            fused_data = pd.merge(fused_data, df, on=sample_key, how='inner', suffixes=('', f'_{name}'))
            
            self.logger.info(f"Après fusion avec {name} : {fused_data.shape}")
        
        # Nettoyer les colonnes en double (si même nom dans différents datasets)
        fused_data = self._clean_duplicate_columns(fused_data)
        
        self.logger.info(f"✅ Fusion horizontale terminée : {fused_data.shape}")
        return fused_data
    
    def vertical_fusion(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Fusion verticale des datasets (concaténation des lignes)
        
        Args:
            datasets: Dictionnaire des datasets à fusionner
        
        Returns:
            DataFrame fusionné
        """
        self.logger.info("Fusion verticale des datasets")
        
        if len(datasets) < 2:
            return list(datasets.values())[0] if datasets else pd.DataFrame()
        
        # Concaténer verticalement
        fused_data = pd.concat(datasets.values(), axis=0, ignore_index=True)
        
        self.logger.info(f"✅ Fusion verticale terminée : {fused_data.shape}")
        return fused_data
    
    def scale_features(self, data: pd.DataFrame, 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Met à l'échelle les features pour la fusion
        
        Args:
            data: Données à scaler
            method: Méthode de scaling ('standard', 'minmax')
        
        Returns:
            Données scalées
        """
        self.logger.info(f"Scaling des features avec méthode : {method}")
        
        # Séparer les colonnes numériques et catégorielles
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("Aucune colonne numérique trouvée pour le scaling")
            return data
        
        # Préparer les données
        result = data.copy()
        numeric_data = result[numeric_cols]
        
        # Appliquer le scaling
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            self.logger.warning(f"Méthode de scaling '{method}' non reconnue")
            return data
        
        # Scaler les données numériques
        scaled_numeric = scaler.fit_transform(numeric_data)
        result[numeric_cols] = scaled_numeric
        
        self.logger.info("✅ Scaling des features terminé")
        return result
    
    def feature_selection(self, data: pd.DataFrame, 
                         target_column: str,
                         method: str = 'variance',
                         threshold: float = 0.1) -> pd.DataFrame:
        """
        Sélection des features les plus pertinentes
        
        Args:
            data: Données avec features et cible
            target_column: Nom de la colonne cible
            method: Méthode de sélection ('variance', 'correlation')
            threshold: Seuil pour la sélection
        
        Returns:
            Données avec features sélectionnées
        """
        self.logger.info(f"Sélection des features par {method}")
        
        if target_column not in data.columns:
            self.logger.error(f"Colonne cible '{target_column}' non trouvée")
            return data
        
        feature_cols = [col for col in data.columns if col != target_column]
        
        if method == 'variance':
            # Sélectionner les features avec variance > threshold
            variances = data[feature_cols].var()
            selected_features = variances[variances > threshold].index.tolist()
            
        elif method == 'correlation':
            # Sélectionner les features avec corrélation absolue > threshold
            correlations = data[feature_cols].corrwith(data[target_column]).abs()
            selected_features = correlations[correlations > threshold].index.tolist()
            
        else:
            self.logger.warning(f"Méthode de sélection '{method}' non reconnue")
            return data
        
        # Conserver la colonne cible et les features sélectionnées
        selected_columns = selected_features + [target_column]
        result = data[selected_columns].copy()
        
        self.logger.info(f"Features sélectionnées : {len(selected_features)}/{len(feature_cols)}")
        return result
    
    def _clean_duplicate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les colonnes en double après fusion"""
        # Trouver les colonnes en double (même nom)
        duplicate_cols = data.columns[data.columns.duplicated()].tolist()
        
        if duplicate_cols:
            self.logger.info(f"Nettoyage des colonnes duplicatas : {duplicate_cols}")
            
            # Pour chaque colonne en double, garder la première et renommer les autres
            for col in duplicate_cols:
                # Compter les occurrences de cette colonne
                col_positions = [i for i, c in enumerate(data.columns) if c == col]
                
                # Renommer les duplicatas (garder le premier)
                for i, pos in enumerate(col_positions[1:], 1):
                    data.columns.values[pos] = f"{col}_dup{i}"
        
        return data

# Tests rapides
if __name__ == "__main__":
    # Créer des datasets de test
    omic_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004'],
        'gene1': [1, 2, 3, 4],
        'gene2': [2, 3, 4, 5]
    })
    
    clinical_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004'],
        'age': [45, 50, 55, 60],
        'stage': ['I', 'II', 'III', 'IV']
    })
    
    print("=== TEST DATA FUSION ===")
    print("Données omiques :")
    print(omic_data)
    print("\nDonnées cliniques :")
    print(clinical_data)
    
    # Test fusion horizontale
    fusion = MultiOmicsFusion(fusion_method='horizontal')
    datasets = {
        'omic': omic_data,
        'clinical': clinical_data
    }
    
    fused = fusion.horizontal_fusion(datasets, sample_key='patient_id')
    
    print("\nAprès fusion horizontale :")
    print(fused)
    print(f"\nDimensions : {fused.shape}")
    
    # Test scaling
    scaled = fusion.scale_features(fused, method='standard')
    print(f"\nAprès scaling :")
    print(scaled)
    print(f"\nStats des colonnes numériques :")
    print(scaled[['gene1', 'gene2', 'age']].describe())