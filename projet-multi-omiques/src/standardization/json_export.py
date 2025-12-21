"""
Module d'export vers format JSON avec schéma - Version simplifiée pour livraison rapide
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class JSONExporter:
    """Exporte les données vers JSON avec schéma standardisé - Version accélérée"""
    
    def __init__(self, schema_version: str = '1.0'):
        self.schema_version = schema_version
        self.logger = logging.getLogger('JSONExporter')
    
    def create_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Crée un schéma JSON pour les données"""
        self.logger.info("Création du schéma JSON")
        
        schema = {
            "schema_version": self.schema_version,
            "created_at": datetime.now().isoformat(),
            "dataset_info": {
                "n_samples": len(data),
                "n_features": len(data.columns),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
                "completeness": 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            },
            "columns": {}
        }
        
        # Décrire chaque colonne
        for col in data.columns:
            col_info = {
                "name": col,
                "dtype": str(data[col].dtype),
                "n_missing": data[col].isnull().sum(),
                "missing_percentage": (data[col].isnull().sum() / len(data)) * 100
            }
            
            # Ajouter des statistiques selon le type
            if pd.api.types.is_numeric_dtype(data[col]):
                col_info.update({
                    "min": float(data[col].min()) if not data[col].isnull().all() else None,
                    "max": float(data[col].max()) if not data[col].isnull().all() else None,
                    "mean": float(data[col].mean()) if not data[col].isnull().all() else None,
                    "median": float(data[col].median()) if not data[col].isnull().all() else None,
                    "std": float(data[col].std()) if not data[col].isnull().all() else None
                })
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                col_info.update({
                    "unique_values": data[col].nunique(),
                    "top_values": data[col].value_counts().head(5).to_dict()
                })
            
            schema["columns"][col] = col_info
        
        return schema
    
    def export_with_schema(self, data: pd.DataFrame, 
                          output_path: str,
                          include_schema: bool = True) -> bool:
        """
        Exporte les données avec leur schéma
        
        Args:
            data: Données à exporter
            output_path: Chemin de sortie
            include_schema: Inclure le schéma dans l'export
        
        Returns:
            True si succès, False sinon
        """
        self.logger.info(f"Export JSON vers {output_path}")
        
        try:
            # Préparer les données pour la sérialisation JSON
            # Convertir les types numpy en types Python natifs
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return obj
            
            # Convertir le DataFrame en dictionnaire
            data_dict = {
                "data": data.map(convert_numpy_types).to_dict('records'),
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "n_samples": len(data),
                    "n_columns": len(data.columns)
                }
            }
            
            # Ajouter le schéma si demandé
            if include_schema:
                data_dict["schema"] = self.create_schema(data)
            
            # Sauvegarder le fichier
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False, 
                         default=convert_numpy_types)
            
            self.logger.info(f"✅ Export JSON terminé : {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'export JSON : {e}")
            return False
    
    def export_simple_json(self, data: pd.DataFrame, output_path: str) -> bool:
        """
        Export simple vers JSON sans schéma
        
        Args:
            data: Données à exporter
            output_path: Chemin de sortie
        
        Returns:
            True si succès, False sinon
        """
        self.logger.info(f"Export JSON simple vers {output_path}")
        
        try:
            # Export simple du DataFrame
            data.to_json(output_path, orient='records', indent=2, force_ascii=False)
            
            self.logger.info(f"✅ Export JSON simple terminé : {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'export JSON simple : {e}")
            return False
    
    def export_split_by_samples(self, data: pd.DataFrame, 
                               output_dir: str,
                               id_column: str = 'patient_id') -> bool:
        """
        Exporte les données avec un fichier JSON par échantillon
        
        Args:
            data: Données à exporter
            output_dir: Répertoire de sortie
            id_column: Colonne contenant les IDs d'échantillon
        
        Returns:
            True si succès, False sinon
        """
        self.logger.info(f"Export par échantillon vers {output_dir}")
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if id_column not in data.columns:
                self.logger.error(f"Colonne ID {id_column} non trouvée")
                return False
            
            exported_count = 0
            for idx, row in data.iterrows():
                sample_id = str(row[id_column])
                
                # Préparer les données de l'échantillon
                sample_data = {
                    "sample_id": sample_id,
                    "data": row.to_dict(),
                    "export_timestamp": datetime.now().isoformat()
                }
                
                # Sauvegarder
                output_path = os.path.join(output_dir, f"{sample_id}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)
                
                exported_count += 1
            
            self.logger.info(f"✅ Export par échantillon terminé : {exported_count} fichiers")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'export par échantillon : {e}")
            return False
    
    def create_api_format(self, data: pd.DataFrame, 
                         endpoint_name: str = "multi_omics_data") -> Dict[str, Any]:
        """
        Crée un format compatible API REST
        
        Args:
            data: Données à formater
            endpoint_name: Nom de l'endpoint API
        
        Returns:
            Structure API formatée
        """
        return {
            "api_version": "1.0",
            "endpoint": endpoint_name,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "samples": len(data),
                "features": len(data.columns),
                "records": data.to_dict('records')
            },
            "metadata": {
                "source": "multi_omics_pipeline",
                "processing_date": datetime.now().strftime("%Y-%m-%d"),
                "format_version": "1.0"
            }
        }
    
    def validate_json_schema(self, data: Dict[str, Any]) -> bool:
        """
        Valide la structure JSON contre le schéma
        
        Args:
            data: Données JSON à valider
        
        Returns:
            True si valide, False sinon
        """
        required_fields = ['data', 'metadata']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Champ requis manquant : {field}")
                return False
        
        if not isinstance(data['data'], list):
            self.logger.error("Le champ 'data' doit être une liste")
            return False
        
        return True

# Tests rapides
if __name__ == "__main__":
    # Créer des données de test
    test_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'gene1': [1.5, 2.3, 0.8],
        'gene2': [2.1, 1.9, 3.2],
        'age': [45, 50, 55],
        'stage': ['I', 'II', 'I']
    })
    
    print("=== TEST JSON EXPORTER ===")
    print("Données originales :")
    print(test_data)
    
    # Test export avec schéma
    exporter = JSONExporter(schema_version='1.0')
    success = exporter.export_with_schema(test_data, 'test_export.json')
    
    print(f"\nExport avec schéma : {'✅ Succès' if success else '❌ Échec'}")
    
    # Test export simple
    success_simple = exporter.export_simple_json(test_data, 'test_simple.json')
    print(f"Export simple : {'✅ Succès' if success_simple else '❌ Échec'}")
    
    # Lire et vérifier le fichier exporté
    try:
        with open('test_export.json', 'r') as f:
            exported_data = json.load(f)
        
        print(f"\nStructure exportée :")
        print(f"- Nombre d'échantillons : {len(exported_data['data'])}")
        print(f"- Avec schéma : {'schema' in exported_data}")
        print(f"- Métadonnées présentes : {'metadata' in exported_data}")
        
        # Validation
        is_valid = exporter.validate_json_schema(exported_data)
        print(f"- Structure valide : {'✅ Oui' if is_valid else '❌ Non'}")
        
    except Exception as e:
        print(f"Erreur lors de la lecture