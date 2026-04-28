#!/usr/bin/env python3
"""
Main multi-omics pipeline
"""
import pandas as pd
import yaml
from pathlib import Path
import logging
from datetime import datetime

class MultiOmicsPipeline:
    """Main pipeline for multi-omics data integration"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initializes the pipeline with the configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            print(f" [OK] Pipeline initialized: {self.config['general']['project_name']}")
            print(f"[*] Version: {self.config['general']['version']}")
            
            # Logging configuration
            self.setup_logging()
            
        except FileNotFoundError:
            print(f" Error: Configuration file '{config_path}' not found")
            raise
        except yaml.YAMLError as e:
            print(f" Error: Invalid YAML configuration - {e}")
            raise
    
    def setup_logging(self):
        """Configures the logging system"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/pipeline.log')
        
        # Create logs directory if necessary
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
        self.logger.info("Logging system configured")
    
    def run(self, omic_data_path, clinical_data_path, output_dir="results"):
        """Executes the complete pipeline"""
        self.logger.info(" Starting multi-omics pipeline")
        
        try:
            # 1. Data loading
            self.logger.info(" Loading data")
            omic_data = self.load_data(omic_data_path)
            clinical_data = self.load_data(clinical_data_path)
            
            self.logger.info(f"Omics data: {omic_data.shape}")
            self.logger.info(f"Clinical data: {clinical_data.shape}")
            
            # 2. Preprocessing
            self.logger.info(" Data preprocessing")
            processed_data = self.preprocess_data(omic_data, clinical_data)
            
            # 3. Integration
            self.logger.info(" Data integration")
            integrated_data = self.integrate_data(processed_data)
            
            # 4. ML Modeling
            self.logger.info(" ML Modeling")
            target = self.config.get('general', {}).get('target_variable', 'treatment_response')
            ml_data, model_results = self.run_model(integrated_data, target_variable=target)
            
            # 5. Export
            self.logger.info(" Results export")
            output_paths = self.export_data(integrated_data, output_dir)
            
            # Export ML dataset if modeling succeeded
            if ml_data is not None:
                self.logger.info(" ML data export")
                from export.ml_exporter import MLExporter
                exporter = MLExporter(config=self.config.get('ml', {}))
                ml_export_results = exporter.save_ml_dataset(ml_data, f"{output_dir}/ml_data")
                output_paths['ml_dataset'] = ml_export_results['output_directory']
            
            self.logger.info(" Pipeline completed successfully")
            
            return {
                'status': 'success',
                'output_paths': output_paths,
                'summary': self.generate_summary(integrated_data),
                'model_results': model_results
            }
            
        except Exception as e:
            self.logger.error(f" Error during execution: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def load_data(self, data_path):
        """Loads data from a CSV file"""
        try:
            data = pd.read_csv(data_path)
            self.logger.info(f" Data loaded from {data_path}: {data.shape}")
            return data
        except Exception as e:
            self.logger.error(f" Error loading {data_path}: {str(e)}")
            raise
    
    def preprocess_data(self, omic_data, clinical_data):
        """Preprocesses data using preprocessing modules"""
        
        # Import preprocessing modules
        from preprocessing.missing_values import MissingValueHandler
        from preprocessing.normalization import OmicsNormalizer
        
        self.logger.info(" Data preprocessing")
        
        # Handle missing values
        self.logger.info("Missing values handling")
        missing_handler = MissingValueHandler(
            strategy=self.config['preprocessing']['missing_values']['strategy'],
            k=self.config['preprocessing']['missing_values'].get('k', 5)
        )
        
        omic_clean = missing_handler.fit_transform(omic_data)
        clinical_clean = missing_handler.fit_transform(clinical_data)
        
        # Normalize omics data
        self.logger.info("Omics data normalization")
        normalizer = OmicsNormalizer(config=self.config)
        
        omic_normalized = normalizer.fit_transform(
            omic_clean,
            data_type='gene_expression',
            method=self.config['preprocessing']['normalization']['method']
        )
        
        # Preprocessing statistics
        preprocessing_info = {
            'omic_missing_values_before': omic_data.isnull().sum().sum(),
            'omic_missing_values_after': omic_clean.isnull().sum().sum(),
            'clinical_missing_values_before': clinical_data.isnull().sum().sum(),
            'clinical_missing_values_after': clinical_clean.isnull().sum().sum(),
            'normalization_method': self.config['preprocessing']['normalization']['method']
        }
        
        self.logger.info(f"Omics missing values: {preprocessing_info['omic_missing_values_before']} -> {preprocessing_info['omic_missing_values_after']}")
        self.logger.info(f"Clinical missing values: {preprocessing_info['clinical_missing_values_before']} -> {preprocessing_info['clinical_missing_values_after']}")
        
        return {
            'omic': omic_normalized,
            'clinical': clinical_clean,
            'preprocessing_info': preprocessing_info
        }
    
    def integrate_data(self, processed_data):
        """Integrates multi-modality data using integration modules"""
        
        # Import integration modules
        from integration.sample_alignment import SampleAlignment
        from integration.data_fusion import MultiOmicsFusion
        
        self.logger.info(" Data integration")
        
        omic_data = processed_data['omic']
        clinical_data = processed_data['clinical']
        
        # Align samples
        self.logger.info("Sample alignment")
        aligner = SampleAlignment(
            fuzzy_matching=self.config['integration']['sample_alignment'].get('fuzzy_matching', False)
        )
        
        aligned_data = aligner.align_by_patient_id(
            {'omic': omic_data, 'clinical': clinical_data},
            {'omic': 'patient_id', 'clinical': 'patient_id'}
        )
        
        # Alignment validation
        validation_report = aligner.validate_alignment(
            {'omic': omic_data, 'clinical': clinical_data}, 
            aligned_data
        )
        
        if not validation_report['alignment_successful']:
            self.logger.error(" Sample alignment failed")
            raise ValueError("Could not align samples")
        
        # Fuse data
        self.logger.info("Multi-modality fusion")
        fusion = MultiOmicsFusion(
            fusion_method=self.config['integration']['data_fusion']['method']
        )
        
        integrated = fusion.horizontal_fusion(aligned_data, sample_key='patient_id')
        
        # Optional feature scaling
        if self.config['integration']['data_fusion'].get('scale_features', False):
            self.logger.info("Feature scaling")
            integrated = fusion.scale_features(integrated, method='standard')
        
        self.logger.info(f" Integration complete: {integrated.shape}")
        return integrated
    
    def run_model(self, integrated_data, target_variable='treatment_response'):
        """Prepares ML data and launches a quick model evaluation"""
        from export.ml_exporter import MLExporter
        
        self.logger.info(" Executing Machine Learning model")
        
        # Basic ML configuration if not present
        ml_config = self.config.get('ml', {
            'test_size': 0.2,
            'random_state': 42,
            'scaling_method': 'standard',
            'feature_selection': {'method': 'mutual_info', 'k_best': 50}
        })
        
        exporter = MLExporter(config=ml_config)
        
        # ML exporter expects a dictionary with 'integrated_data' key
        data_dict = {'integrated_data': integrated_data}
        
        try:
            # Check if target variable exists
            if target_variable not in integrated_data.columns:
                self.logger.warning(f"Target variable '{target_variable}' not found. Modeling skipped.")
                return None, {'error': f"Target variable '{target_variable}' not found"}
                
            # 1. Prepare ML data
            self.logger.info("Preparing data for ML...")
            ml_data = exporter.prepare_ml_data(data_dict, target_variable=target_variable)
            
            # 2. Evaluate model
            self.logger.info("Evaluating model (Random Forest)...")
            eval_results = exporter.quick_model_evaluation(ml_data, model_type='random_forest')
            
            if 'test_accuracy' in eval_results:
                self.logger.info(f" Model evaluated. Test accuracy: {eval_results.get('test_accuracy', 0):.3f}")
            else:
                self.logger.warning(f"Model evaluated but accuracy not available: {eval_results.get('error', 'Unknown error')}")
                
            return ml_data, eval_results
            
        except Exception as e:
            self.logger.error(f" Error during model execution: {str(e)}")
            return None, {'error': str(e)}
    
    def export_data(self, integrated_data, output_dir):
        """Exports data in different formats using export modules"""
        
        # Import export modules
        from standardization.json_export import JSONExporter
        from standardization.csv_export import CSVExporter
        
        Path(output_dir).mkdir(exist_ok=True)
        
        output_paths = {}
        
        # Standardized CSV export
        self.logger.info(" CSV Export")
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
            self.logger.info(f" Data exported to CSV: {csv_path}")
        
        # JSON export with schema
        self.logger.info(" JSON Export")
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
            self.logger.info(f" Data exported to JSON: {json_path}")
        
        # OPTIONAL FHIR (if time permits)
        if 'fhir' in self.config['export'] and self.config['export']['fhir'].get('enabled', False):
            self.logger.info(" FHIR Export (optional)")
            # TODO: Implement FHIR export if time available
            pass
        
        return output_paths
    
    def generate_summary(self, data):
        """Generates a summary of the processed data"""
        return {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024*1024),
            'completeness': 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
        }

# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Omics Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--omic-data", required=True, help="Path to omics data")
    parser.add_argument("--clinical-data", required=True, help="Path to clinical data")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"[OK] Multi-Omics Pipeline - Version 1.0")
    print("=" * 50)
    
    pipeline = MultiOmicsPipeline(args.config)
    result = pipeline.run(args.omic_data, args.clinical_data, args.output_dir)
    
    print("\n" + "=" * 50)
    print(" Pipeline results:")
    
    if result['status'] == 'success':
        print(" Pipeline executed successfully!")
        print(f"[*] Output files: {result['output_paths']}")
        print(f"[*] Summary: {result['summary']}")
        
        if 'model_results' in result and 'test_accuracy' in result['model_results']:
            acc = result['model_results']['test_accuracy']
            print(f" ML model accuracy: {acc:.3f}")
    else:
        print(f" Error: {result['error']}")