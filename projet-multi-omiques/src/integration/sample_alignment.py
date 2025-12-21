"""
Module d'alignement des échantillons - Version simplifiée pour livraison rapide
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

class SampleAlignment:
    """Aligne les échantillons entre différentes modalités de données - Version accélérée"""
    
    def __init__(self, fuzzy_matching: bool = False, threshold: float = 0.9):
        self.fuzzy_matching = fuzzy_matching
        self.threshold = threshold
        self.logger = logging.getLogger('SampleAlignment')
    
    def align_by_patient_id(self, datasets: Dict[str, pd.DataFrame], 
                          patient_id_columns: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Aligne les datasets par ID patient - Version simplifiée
        
        Args:
            datasets: Dictionnaire des datasets à aligner
            patient_id_columns: Dictionnaire des colonnes ID patient pour chaque dataset
        
        Returns:
            Dictionnaire des datasets alignés
        """
        self.logger.info("Alignement des échantillons par ID patient")
        
        if len(datasets) < 2:
            self.logger.warning("Moins de 2 datasets à aligner")
            return datasets
        
        # Obtenir la liste des IDs patients pour chaque dataset
        patient_ids = {}
        for name, df in datasets.items():
            if name not in patient_id_columns:
                self.logger.warning(f"Colonne ID patient non spécifiée pour {name}")
                continue
            
            id_col = patient_id_columns[name]
            if id_col not in df.columns:
                self.logger.error(f"Colonne {id_col} non trouvée dans {name}")
                continue
            
            patient_ids[name] = set(df[id_col].dropna().astype(str))
        
        if len(patient_ids) < 2:
            self.logger.error("Impossible d'aligner - pas assez de datasets avec IDs")
            return datasets
        
        # Trouver les IDs patients communs
        common_ids = set.intersection(*patient_ids.values())
        self.logger.info(f"IDs patients communs trouvés : {len(common_ids)}")
        
        if len(common_ids) == 0:
            self.logger.warning("Aucun ID patient commun trouvé")
            return datasets
        
        # Filtrer chaque dataset pour garder seulement les échantillons communs
        aligned_datasets = {}
        for name, df in datasets.items():
            if name not in patient_id_columns:
                aligned_datasets[name] = df
                continue
            
            id_col = patient_id_columns[name]
            aligned_df = df[df[id_col].astype(str).isin(common_ids)].copy()
            aligned_datasets[name] = aligned_df
            
            self.logger.info(f"{name} : {len(df)} → {len(aligned_df)} échantillons")
        
        return aligned_datasets
    
    def align_by_metadata(self, datasets: Dict[str, pd.DataFrame], 
                         metadata_keys: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Aligne les datasets par métadonnées - Version simplifiée
        
        Args:
            datasets: Dictionnaire des datasets à aligner
            metadata_keys: Liste des clés de métadonnées pour l'alignement
        
        Returns:
            Dictionnaire des datasets alignés
        """
        self.logger.info(f"Alignement par métadonnées : {metadata_keys}")
        
        if len(datasets) < 2:
            return datasets
        
        # Vérifier que toutes les clés de métadonnées existent dans tous les datasets
        for name, df in datasets.items():
            for key in metadata_keys:
                if key not in df.columns:
                    self.logger.error(f"Clé de métadonnée {key} manquante dans {name}")
                    return datasets
        
        # Créer des clés de hachage pour chaque échantillon
        dataset_keys = {}
        for name, df in datasets.items():
            # Créer une clé combinée pour chaque échantillon
            keys = df[metadata_keys].astype(str).apply(lambda x: '_'.join(x), axis=1)
            dataset_keys[name] = set(keys.dropna())
        
        # Trouver les clés communes
        common_keys = set.intersection(*dataset_keys.values())
        self.logger.info(f"Clés de métadonnées communes : {len(common_keys)}")
        
        if len(common_keys) == 0:
            return datasets
        
        # Filtrer les datasets
        aligned_datasets = {}
        for name, df in datasets.items():
            keys = df[metadata_keys].astype(str).apply(lambda x: '_'.join(x), axis=1)
            aligned_df = df[keys.isin(common_keys)].copy()
            aligned_datasets[name] = aligned_df
            
            self.logger.info(f"{name} : {len(df)} → {len(aligned_df)} échantillons")
        
        return aligned_datasets
    
    def validate_alignment(self, original_data: Dict[str, pd.DataFrame], 
                          aligned_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Valide l'alignement des données
        
        Args:
            original_data: Données originales
            aligned_data: Données alignées
        
        Returns:
            Rapport de validation
        """
        self.logger.info("Validation de l'alignement")
        
        report = {
            'alignment_successful': True,
            'datasets_validated': [],
            'total_samples_lost': 0,
            'warnings': [],
            'errors': []
        }
        
        for name in original_data.keys():
            if name not in aligned_data:
                report['errors'].append(f"Dataset {name} manquant après alignement")
                report['alignment_successful'] = False
                continue
            
            original_count = len(original_data[name])
            aligned_count = len(aligned_data[name])
            samples_lost = original_count - aligned_count
            
            report['datasets_validated'].append({
                'name': name,
                'original_count': original_count,
                'aligned_count': aligned_count,
                'samples_lost': samples_lost,
                'retention_rate': aligned_count / original_count if original_count > 0 else 0
            })
            
            report['total_samples_lost'] += samples_lost
            
            if samples_lost > 0:
                report['warnings'].append(f"{name} : {samples_lost} échantillons perdus")
            
            if aligned_count == 0:
                report['errors'].append(f"{name} : aucun échantillon conservé")
                report['alignment_successful'] = False
        
        # Vérifier que tous les datasets ont le même nombre d'échantillons
        aligned_counts = [len(aligned_data[name]) for name in aligned_data.keys()]
        if len(set(aligned_counts)) > 1:
            report['errors'].append("Les datasets n'ont pas le même nombre d'échantillons")
            report['alignment_successful'] = False
        
        self.logger.info(f"Validation terminée : {'Succès' if report['alignment_successful'] else 'Échec'}")
        
        return report
    
    def handle_missing_samples(self, aligned_data: Dict[str, pd.DataFrame], 
                             patient_id_columns: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Gère les échantillons manquants après alignement
        
        Args:
            aligned_data: Données alignées
            patient_id_columns: Colonnes ID patient
        
        Returns:
            Données avec gestion des manquants
        """
        self.logger.info("Gestion des échantillons manquants")
        
        # Pour cette version simplifiée, on ne fait pas de gestion complexe
        # On retourne les données telles quelles
        
        return aligned_data

# Tests rapides
if __name__ == "__main__":
    # Créer des datasets de test
    omic_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'gene1': [1, 2, 3, 4, 5],
        'gene2': [2, 3, 4, 5, 6]
    })
    
    clinical_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P004', 'P005', 'P006'],
        'age': [45, 50, 55, 60, 65],
        'stage': ['I', 'II', 'III', 'IV', 'I']
    })
    
    print("=== TEST SAMPLE ALIGNMENT ===")
    print("Données omiques :")
    print(omic_data)
    print("\nDonnées cliniques :")
    print(clinical_data)
    
    # Test alignement
    aligner = SampleAlignment()
    datasets = {
        'omic': omic_data,
        'clinical': clinical_data
    }
    
    aligned = aligner.align_by_patient_id(
        datasets,
        {'omic': 'patient_id', 'clinical': 'patient_id'}
    )
    
    print("\nAprès alignement :")
    for name, df in aligned.items():
        print(f"\n{name} :")
        print(df)
    
    # Validation
    report = aligner.validate_alignment(datasets, aligned)
    print(f"\nRapport de validation : {report}")