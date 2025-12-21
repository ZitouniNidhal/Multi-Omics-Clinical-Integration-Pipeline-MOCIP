"""
Module de collecte des données TCGA via GDC API
"""
import pandas as pd
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

class TCGADataCollector:
    """Collecte les données TCGA via GDC API"""
    
    def __init__(self, project_id: str = "TCGA-BRCA"):
        self.project_id = project_id
        self.base_url = "https://api.gdc.cancer.gov"
        self.session = requests.Session()
        self.logger = logging.getLogger('TCGADataCollector')
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
    
    def search_files(self, 
                    data_category: str = "Transcriptome Profiling",
                    data_type: str = "Gene Expression Quantification",
                    workflow_type: str = "HTSeq - Counts") -> List[str]:
        """
        Recherche les fichiers de données dans TCGA
        
        Args:
            data_category: Catégorie de données (ex: "Transcriptome Profiling")
            data_type: Type de données (ex: "Gene Expression Quantification")
            workflow_type: Type de workflow (ex: "HTSeq - Counts")
        
        Returns:
            Liste des IDs de fichiers
        """
        self.logger.info(f"Recherche de fichiers pour {self.project_id}")
        
        endpoint = f"{self.base_url}/files"
        
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [self.project_id]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_category",
                        "value": [data_category]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": [data_type]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": [workflow_type]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.case_id",
            "size": 2000,
            "format": "JSON"
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', {}).get('hits', [])
            
            file_info = []
            for file in files:
                file_info.append({
                    'file_id': file['file_id'],
                    'file_name': file['file_name'],
                    'case_id': file['cases'][0]['case_id']
                })
            
            self.logger.info(f"Trouvé {len(file_info)} fichiers")
            return file_info
            
        except requests.RequestException as e:
            self.logger.error(f"Erreur lors de la recherche : {e}")
            return []
    
    def download_file(self, file_id: str, output_path: str) -> bool:
        """
        Télécharge un fichier depuis TCGA
        
        Args:
            file_id: ID du fichier GDC
            output_path: Chemin de sortie
        
        Returns:
            True si succès, False sinon
        """
        self.logger.info(f"Téléchargement du fichier {file_id}")
        
        endpoint = f"{self.base_url}/data/{file_id}"
        
        try:
            response = self.session.get(endpoint, stream=True)
            response.raise_for_status()
            
            # Créer le répertoire de sortie
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Télécharger le fichier
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"✅ Fichier téléchargé : {output_path}")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"❌ Erreur lors du téléchargement : {e}")
            return False
    
    def get_clinical_data(self, case_ids: List[str]) -> pd.DataFrame:
        """
        Récupère les données cliniques pour une liste de cas
        
        Args:
            case_ids: Liste des IDs de cas
        
        Returns:
            DataFrame avec les données cliniques
        """
        self.logger.info(f"Récupération des données cliniques pour {len(case_ids)} cas")
        
        endpoint = f"{self.base_url}/cases"
        
        filters = {
            "op": "in",
            "content": {
                "field": "case_id",
                "value": case_ids
            }
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "case_id,demographic,diagnoses,treatments",
            "size": len(case_ids),
            "format": "JSON"
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            cases = data.get('data', {}).get('hits', [])
            
            clinical_data = []
            for case in cases:
                record = {
                    'case_id': case['case_id']
                }
                
                # Informations démographiques
                if 'demographic' in case:
                    demo = case['demographic']
                    record.update({
                        'age_at_diagnosis': demo.get('age_at_index'),
                        'gender': demo.get('gender'),
                        'race': demo.get('race'),
                        'ethnicity': demo.get('ethnicity')
                    })
                
                # Informations sur le diagnostic
                if 'diagnoses' in case and case['diagnoses']:
                    diagnosis = case['diagnoses'][0]
                    record.update({
                        'primary_diagnosis': diagnosis.get('primary_diagnosis'),
                        'tumor_stage': diagnosis.get('ajcc_pathologic_stage'),
                        'tumor_grade': diagnosis.get('ajcc_pathologic_grade'),
                        'histological_type': diagnosis.get('histological_type')
                    })
                
                clinical_data.append(record)
            
            df = pd.DataFrame(clinical_data)
            self.logger.info(f"✅ Données cliniques récupérées : {df.shape}")
            return df
            
        except requests.RequestException as e:
            self.logger.error(f"❌ Erreur lors de la récupération des données cliniques : {e}")
            return pd.DataFrame()
    
    def download_expression_matrix(self, 
                                 output_dir: str,
                                 max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Télécharge les fichiers d'expression et construit une matrice
        
        Args:
            output_dir: Répertoire de sortie
            max_files: Nombre maximum de fichiers à télécharger
        
        Returns:
            DataFrame avec la matrice d'expression
        """
        self.logger.info("Construction de la matrice d'expression")
        
        # Rechercher les fichiers
        files = self.search_files()
        
        if not files:
            self.logger.warning("Aucun fichier trouvé")
            return pd.DataFrame()
        
        # Limiter le nombre de fichiers si demandé
        if max_files:
            files = files[:max_files]
            self.logger.info(f"Limité à {max_files} fichiers")
        
        # Télécharger les fichiers et construire la matrice
        expression_data = {}
        case_ids = []
        
        for i, file_info in enumerate(files):
            file_id = file_info['file_id']
            case_id = file_info['case_id']
            file_name = file_info['file_name']
            
            self.logger.info(f"Processing file {i+1}/{len(files)}: {file_name}")
            
            # Télécharger le fichier
            file_path = f"{output_dir}/raw/{file_name}"
            if self.download_file(file_id, file_path):
                try:
                    # Lire le fichier d'expression (format HTSeq counts)
                    df = pd.read_csv(file_path, sep='\t', header=None, names=['gene_id', 'count'])
                    df['gene_id'] = df['gene_id'].str.split('.').str[0]  # Enlever la version
                    
                    # Stocker les données
                    expression_data[case_id] = df.set_index('gene_id')['count']
                    case_ids.append(case_id)
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de la lecture de {file_name}: {e}")
            
            # Petite pause pour éviter de surcharger l'API
            time.sleep(0.5)
        
        # Construire la matrice d'expression
        if expression_data:
            expression_matrix = pd.DataFrame(expression_data).T
            expression_matrix.index.name = 'case_id'
            
            # Sauvegarder la matrice
            output_file = f"{output_dir}/expression_matrix.csv"
            expression_matrix.to_csv(output_file)
            self.logger.info(f"✅ Matrice d'expression sauvegardée : {output_file}")
            
            return expression_matrix
        
        return pd.DataFrame()
    
    def download_dataset(self, 
                        output_dir: str,
                        max_files: Optional[int] = None,
                        download_clinical: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Télécharge un jeu de données complet
        
        Args:
            output_dir: Répertoire de sortie
            max_files: Nombre maximum de fichiers
            download_clinical: Télécharger aussi les données cliniques
        
        Returns:
            Dictionnaire avec les données téléchargées
        """
        self.logger.info(f"Téléchargement du jeu de données {self.project_id}")
        
        results = {}
        
        # Télécharger la matrice d'expression
        expression_matrix = self.download_expression_matrix(output_dir, max_files)
        results['expression'] = expression_matrix
        
        # Télécharger les données cliniques
        if download_clinical and not expression_matrix.empty:
            case_ids = expression_matrix.index.tolist()
            clinical_data = self.get_clinical_data(case_ids)
            
            if not clinical_data.empty:
                clinical_file = f"{output_dir}/clinical_data.csv"
                clinical_data.to_csv(clinical_file, index=False)
                results['clinical'] = clinical_data
                self.logger.info(f"✅ Données cliniques sauvegardées : {clinical_file}")
        
        self.logger.info("✅ Téléchargement terminé")
        return results

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le collecteur
    collector = TCGADataCollector(project_id="TCGA-BRCA")
    
    # Télécharger un petit jeu de données pour tester
    results = collector.download_dataset(
        output_dir="data/raw",
        max_files=5,  # Limiter pour le test
        download_clinical=True
    )
    
    print(f"Données téléchargées : {list(results.keys())}")