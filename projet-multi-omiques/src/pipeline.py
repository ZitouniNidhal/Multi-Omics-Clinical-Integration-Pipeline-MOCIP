#!/usr/bin/env python3
"""
Pipeline principal multi-omiques
"""
import pandas as pd
import yaml
from pathlib import Path
import logging
from datetime import datetime

class MultiOmicsPipeline:
    """Pipeline principal pour l'intégration de données multi-omiques"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialise le pipeline avec la configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            print(f"✅ Pipeline initialisé : {self.config['general']['project_name']}")
            print(f"📋 Version : {self.config['general']['version']}")
            
            # Configuration du logging
            self.setup_logging()
            
        except FileNotFoundError:
            print(f"❌ Erreur : Fichier de configuration '{config_path}' non trouvé")
            raise
        except yaml.YAMLError as e:
            print(f"❌ Erreur : Configuration YAML invalide - {e}")
            raise
    
    def setup_logging(self):
        """Configure le système de logging"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/pipeline.log')
        
        # Créer le répertoire logs si nécessaire
        Path(log_file).parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('MultiOmicsPipeline')
        self.logger.info("Système de logging configuré")
    
    def run(self, omic_data_path, clinical_data_path, output_dir="results"):
        """Exécute le pipeline complet"""
        self.logger.info("🚀 Démarrage du pipeline multi-omiques")
        
        try:
            # 1. Chargement des données
            self.logger.info("📊 Chargement des données")
            omic_data = self.load_data(omic_data_path)
            clinical_data = self.load_data(clinical_data_path)
            
            self.logger.info(f"Données omiques : {omic_data.shape}")
            self.logger.info(f"Données cliniques : {clinical_data.shape}")
            
            # 2. Prétraitement
            self.logger.info("🔧 Prétraitement des données")
            processed_data = self.preprocess_data(omic_data, clinical_data)
            
            # 3. Intégration
            self.logger.info("🔗 Intégration des données")
            integrated_data = self.integrate_data(processed_data)
            
            # 4. Modélisation ML
            self.logger.info("🤖 Modélisation ML")
            target = self.config.get('general', {}).get('target_variable', 'treatment_response')
            ml_data, model_results = self.run_model(integrated_data, target_variable=target)
            
            # 5. Export
            self.logger.info("📤 Export des résultats")
            output_paths = self.export_data(integrated_data, output_dir)
            
            # Export du dataset ML si la modélisation a réussi
            if ml_data is not None:
                self.logger.info("📤 Export des données ML")
                from .export.ml_exporter import MLExporter
                exporter = MLExporter(config=self.config.get('ml', {}))
                ml_export_results = exporter.save_ml_dataset(ml_data, f"{output_dir}/ml_data")
                output_paths['ml_dataset'] = ml_export_results['output_directory']
            
            self.logger.info("✅ Pipeline terminé avec succès")
            
            return {
                'status': 'success',
                'output_paths': output_paths,
                'summary': self.generate_summary(integrated_data),
                'model_results': model_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'exécution : {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def load_data(self, data_path):
        """Charge les données depuis un fichier CSV"""
        try:
            data = pd.read_csv(data_path)
            self.logger.info(f"✅ Données chargées depuis {data_path} : {data.shape}")
            return data
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement de {data_path} : {str(e)}")
            raise
    
    def preprocess_data(self, omic_data, clinical_data):
        """Prétraite les données avec les modules de préprocessing"""
        
        # Importer les modules de préprocessing
        from .preprocessing.missing_values import MissingValueHandler
        from .preprocessing.normalization import OmicsNormalizer
        
        self.logger.info("🔧 Prétraitement des données")
        
        # Gérer les valeurs manquantes
        self.logger.info("Gestion des valeurs manquantes")
        missing_handler = MissingValueHandler(
            strategy=self.config['preprocessing']['missing_values']['strategy'],
            k=self.config['preprocessing']['missing_values'].get('k', 5)
        )
        
        omic_clean = missing_handler.fit_transform(omic_data)
        clinical_clean = missing_handler.fit_transform(clinical_data)
        
        # Normaliser les données omiques
        self.logger.info("Normalisation des données omiques")
        normalizer = OmicsNormalizer(
            method=self.config['preprocessing']['normalization']['method']
        )
        
        omic_normalized = normalizer.normalize(omic_clean)
        
        # Statistiques de préprocessing
        preprocessing_info = {
            'omic_missing_values_before': omic_data.isnull().sum().sum(),
            'omic_missing_values_after': omic_clean.isnull().sum().sum(),
            'clinical_missing_values_before': clinical_data.isnull().sum().sum(),
            'clinical_missing_values_after': clinical_clean.isnull().sum().sum(),
            'normalization_method': self.config['preprocessing']['normalization']['method']
        }
        
        self.logger.info(f"Valeurs manquantes omiques : {preprocessing_info['omic_missing_values_before']} → {preprocessing_info['omic_missing_values_after']}")
        self.logger.info(f"Valeurs manquantes cliniques : {preprocessing_info['clinical_missing_values_before']} → {preprocessing_info['clinical_missing_values_after']}")
        
        return {
            'omic': omic_normalized,
            'clinical': clinical_clean,
            'preprocessing_info': preprocessing_info
        }
    
    def integrate_data(self, processed_data):
        """Intègre les données multi-modalités avec les modules d'intégration"""
        
        # Importer les modules d'intégration
        from .integration.sample_alignment import SampleAlignment
        from .integration.data_fusion import MultiOmicsFusion
        
        self.logger.info("🔗 Intégration des données")
        
        omic_data = processed_data['omic']
        clinical_data = processed_data['clinical']
        
        # Aligner les échantillons
        self.logger.info("Alignement des échantillons")
        aligner = SampleAlignment(
            fuzzy_matching=self.config['integration']['sample_alignment'].get('fuzzy_matching', False)
        )
        
        aligned_data = aligner.align_by_patient_id(
            {'omic': omic_data, 'clinical': clinical_data},
            {'omic': 'patient_id', 'clinical': 'patient_id'}
        )
        
        # Validation de l'alignement
        validation_report = aligner.validate_alignment(
            {'omic': omic_data, 'clinical': clinical_data}, 
            aligned_data
        )
        
        if not validation_report['alignment_successful']:
            self.logger.error("❌ Échec de l'alignement des échantillons")
            raise ValueError("Impossible d'aligner les échantillons")
        
        # Fusionner les données
        self.logger.info("Fusion multi-modalités")
        fusion = MultiOmicsFusion(
            fusion_method=self.config['integration']['data_fusion']['method']
        )
        
        integrated = fusion.horizontal_fusion(aligned_data, sample_key='patient_id')
        
        # Scaling optionnel des features
        if self.config['integration']['data_fusion'].get('scale_features', False):
            self.logger.info("Scaling des features")
            integrated = fusion.scale_features(integrated, method='standard')
        
        self.logger.info(f"✅ Intégration terminée : {integrated.shape}")
        return integrated
    
    def run_model(self, integrated_data, target_variable='treatment_response'):
        """Prépare les données ML et lance une évaluation rapide du modèle"""
        from .export.ml_exporter import MLExporter
        
        self.logger.info("🤖 Exécution du modèle de Machine Learning")
        
        # Configuration ML de base si non présente
        ml_config = self.config.get('ml', {
            'test_size': 0.2,
            'random_state': 42,
            'scaling_method': 'standard',
            'feature_selection': {'method': 'mutual_info', 'k_best': 50}
        })
        
        exporter = MLExporter(config=ml_config)
        
        # L'exportateur ML attend un dictionnaire avec la clé 'integrated_data'
        data_dict = {'integrated_data': integrated_data}
        
        try:
            # Vérifier si la variable cible existe
            if target_variable not in integrated_data.columns:
                self.logger.warning(f"⚠️ Variable cible '{target_variable}' introuvable. Modélisation ignorée.")
                return None, {'error': f"Target variable '{target_variable}' not found"}
                
            # 1. Préparer les données ML
            self.logger.info("Préparation des données pour le ML...")
            ml_data = exporter.prepare_ml_data(data_dict, target_variable=target_variable)
            
            # 2. Évaluer le modèle
            self.logger.info("Évaluation du modèle (Random Forest)...")
            eval_results = exporter.quick_model_evaluation(ml_data, model_type='random_forest')
            
            if 'test_accuracy' in eval_results:
                self.logger.info(f"✅ Modèle évalué. Précision test : {eval_results.get('test_accuracy', 0):.3f}")
            else:
                self.logger.warning(f"⚠️ Modèle évalué mais précision non disponible: {eval_results.get('error', 'Erreur inconnue')}")
                
            return ml_data, eval_results
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'exécution du modèle : {str(e)}")
            return None, {'error': str(e)}
    
    def export_data(self, integrated_data, output_dir):
        """Exporte les données dans différents formats avec les modules d'export"""
        
        # Importer les modules d'export
        from .standardization.json_export import JSONExporter
        from .standardization.csv_export import CSVExporter
        
        Path(output_dir).mkdir(exist_ok=True)
        
        output_paths = {}
        
        # Export CSV standardisé
        self.logger.info("📤 Export CSV")
        csv_exporter = CSVExporter(
            separator=self.config['export']['csv'].get('separator', '\t'),
            include_header=self.config['export']['csv'].get('include_header', True)
        )
        
        csv_path = f"{output_dir}/integrated_data.csv"
        csv_success = csv_exporter.export_standard_csv(
            integrated_data, 
            csv_path,
            {
                'pipeline_version': self.config['general']['version'],
                'export_date': datetime.now().isoformat(),
                'n_samples': len(integrated_data),
                'n_features': len(integrated_data.columns)
            }
        )
        
        if csv_success:
            output_paths['csv'] = csv_path
            self.logger.info(f"✅ Données exportées en CSV : {csv_path}")
        
        # Export JSON avec schéma
        self.logger.info("📤 Export JSON")
        json_exporter = JSONExporter(
            schema_version=self.config['export']['json']['schema_version']
        )
        
        json_path = f"{output_dir}/integrated_data.json"
        json_success = json_exporter.export_with_schema(
            integrated_data, 
            json_path,
            include_metadata=self.config['export']['json'].get('include_metadata', True)
        )
        
        if json_success:
            output_paths['json'] = json_path
            self.logger.info(f"✅ Données exportées en JSON : {json_path}")
        
        # FHIR OPTIONNEL (si temps disponible)
        if 'fhir' in self.config['export'] and self.config['export']['fhir'].get('enabled', False):
            self.logger.info("📤 Export FHIR (optionnel)")
            # TODO: Implémenter FHIR export si temps disponible
            pass
        
        return output_paths
    
    def generate_summary(self, data):
        """Génère un résumé des données traitées"""
        return {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024*1024),
            'completeness': 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
        }

# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Multi-Omiques")
    parser.add_argument("--config", default="config/config.yaml", help="Chemin du fichier de configuration")
    parser.add_argument("--omic-data", required=True, help="Chemin des données omiques")
    parser.add_argument("--clinical-data", required=True, help="Chemin des données cliniques")
    parser.add_argument("--output-dir", default="results", help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    print(f"🧬 Pipeline Multi-Omiques - Version 1.0")
    print("=" * 50)
    
    pipeline = MultiOmicsPipeline(args.config)
    result = pipeline.run(args.omic_data, args.clinical_data, args.output_dir)
    
    print("\\n" + "=" * 50)
    print("📊 Résultats du pipeline :")
    
    if result['status'] == 'success':
        print("✅ Pipeline exécuté avec succès!")
        print(f"📁 Fichiers de sortie : {result['output_paths']}")
        print(f"📈 Résumé : {result['summary']}")
        
        if 'model_results' in result and 'test_accuracy' in result['model_results']:
            acc = result['model_results']['test_accuracy']
            print(f"🤖 Précision du modèle ML : {acc:.3f}")
    else:
        print(f"❌ Erreur : {result['error']}")