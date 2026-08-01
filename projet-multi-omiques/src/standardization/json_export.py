"""
Module for exporting to JSON format with schema - Simplified version for rapid delivery
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class JSONExporter:
    """Exports data to JSON with standardized schema - Accelerated version Ratio"""
    
    def __init__(self, schema_version: str = '1.0'):
        self.schema_version = schema_version
        self.logger = logging.getLogger('JSONExporter')
    
    def create_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Creates a JSON schema for the data"""
        self.logger.info("Creating JSON schema")
        
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
        
        # Describe each column
        for col in data.columns:
            col_info = {
                "name": col,
                "dtype": str(data[col].dtype),
                "n_missing": data[col].isnull().sum(),
                "missing_percentage": (data[col].isnull().sum() / len(data)) * 100
            }
            
            # Add statistics depending on type
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
                          include_schema: bool = True, **kwargs) -> bool:
        """
        Exports the data with its schema
        
        Args:
            data: Data to export
            output_path: Output path
            include_schema: Include the schema in the export
        
        Returns:
            True if success, False otherwise
        """
        self.logger.info(f"Exporting JSON to {output_path}")
        
        try:
            # Prepare data for JSON serialization
            # Convert numpy types to native Python types
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
            
            # Convert DataFrame to dictionary
            data_dict = {
                "data": data.map(convert_numpy_types).to_dict('records'),
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "n_samples": len(data),
                    "n_columns": len(data.columns)
                }
            }
            
            # Add schema if requested
            if include_schema:
                data_dict["schema"] = self.create_schema(data)
            
            # Save the file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False, 
                         default=convert_numpy_types)
            
            self.logger.info(f"[OK] JSON export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[Error] Error during JSON export: {e}")
            return False
    
    def export_simple_json(self, data: pd.DataFrame, output_path: str) -> bool:
        """
        Simple export to JSON without schema
        
        Args:
            data: Data to export
            output_path: Output path
        
        Returns:
            True if success, False otherwise
        """
        self.logger.info(f"Exporting simple JSON to {output_path}")
        
        try:
            # Simple export of the DataFrame
            data.to_json(output_path, orient='records', indent=2, force_ascii=False)
            
            self.logger.info(f"[OK] Simple JSON export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[Error] Error during simple JSON export: {e}")
            return False
    
    def export_split_by_samples(self, data: pd.DataFrame, 
                               output_dir: str,
                               id_column: str = 'patient_id') -> bool:
        """
        Exports data with one JSON file per sample
        
        Args:
            data: Data to export
            output_dir: Output directory
            id_column: Column containing sample IDs
        
        Returns:
            True if success, False otherwise
        """
        self.logger.info(f"Export by sample to {output_dir}")
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if id_column not in data.columns:
                self.logger.error(f"ID column {id_column} not found")
                return False
            
            exported_count = 0
            for idx, row in data.iterrows():
                sample_id = str(row[id_column])
                
                # Prepare sample data
                sample_data = {
                    "sample_id": sample_id,
                    "data": row.to_dict(),
                    "export_timestamp": datetime.now().isoformat()
                }
                
                # Save
                output_path = os.path.join(output_dir, f"{sample_id}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)
                
                exported_count += 1
            
            self.logger.info(f"[OK] Export by sample completed: {exported_count} files")
            return True
            
        except Exception as e:
            self.logger.error(f"[Error] Error during export by sample: {e}")
            return False
    
    def create_api_format(self, data: pd.DataFrame, 
                         endpoint_name: str = "multi_omics_data") -> Dict[str, Any]:
        """
        Creates a format compatible with REST API
        
        Args:
            data: Data to format
            endpoint_name: API endpoint name
        
        Returns:
            Formatted API structure
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
        Validates JSON structure against the schema
        
        Args:
            data: JSON data to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['data', 'metadata']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Required field missing: {field}")
                return False
        
        if not isinstance(data['data'], list):
            self.logger.error("Field 'data' must be a list")
            return False
        
        return True

# Quick tests
if __name__ == "__main__":
    # Create test data
    test_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'gene1': [1.5, 2.3, 0.8],
        'gene2': [2.1, 1.9, 3.2],
        'age': [45, 50, 55],
        'stage': ['I', 'II', 'I']
    })
    
    print("=== TEST JSON EXPORTER ===")
    print("Original data:")
    print(test_data)
    
    # Test export with schema
    exporter = JSONExporter(schema_version='1.0')
    success = exporter.export_with_schema(test_data, 'test_export.json')
    
    print(f"\nExport with schema: {'[OK] Success' if success else '[Error] Failure'}")
    
    # Test simple export
    success_simple = exporter.export_simple_json(test_data, 'test_simple.json')
    print(f"Simple export: {'[OK] Success' if success_simple else '[Error] Failure'}")
    
    # Read and verify the exported file
    try:
        with open('test_export.json', 'r') as f:
            exported_data = json.load(f)
        
        print(f"\nExported structure:")
        print(f"- Number of samples: {len(exported_data['data'])}")
        print(f"- With schema: {'schema' in exported_data}")
        print(f"- Metadata present: {'metadata' in exported_data}")
        
        # Validation
        is_valid = exporter.validate_json_schema(exported_data)
        print(f"- Valid structure: {'[OK] Yes' if is_valid else '[Error] No'}")
        
    except Exception as e:
        print(f"Error during reading: {e}")
