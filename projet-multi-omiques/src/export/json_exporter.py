"""JSON exporter for multi-omics data."""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import logging
from ..exceptions import ExportError

logger = logging.getLogger(__name__)


class JSONExporter:
    """Export multi-omics data to JSON format."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize JSON exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.schema_validation = self.config.get('schema_validation', True)
        self.compact_format = self.config.get('compact_format', False)
        self.include_metadata = self.config.get('include_metadata', True)
        
        logger.info("Initialized JSONExporter")
    
    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               output_file: str, schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Export data to JSON format.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            output_file: Output file path
            schema: Optional JSON schema for validation
            **kwargs: Additional parameters
            
        Returns:
            Export results
        """
        logger.info(f"Exporting data to JSON format: {output_file}")
        
        try:
            # Convert data to JSON-serializable format
            if isinstance(data, pd.DataFrame):
                json_data = self._dataframe_to_json(data, **kwargs)
            elif isinstance(data, dict):
                json_data = {}
                for key, df in data.items():
                    json_data[key] = self._dataframe_to_json(df, **kwargs)
            else:
                raise ExportError(f"Unsupported data type: {type(data)}")
            
            # Add metadata if requested
            if self.include_metadata:
                json_data = self._add_metadata(json_data, data, **kwargs)
            
            # Validate against schema if provided
            if self.schema_validation and schema:
                self._validate_json_schema(json_data, schema)
            
            # Save to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if self.compact_format:
                    json.dump(json_data, f, separators=(',', ':'))
                else:
                    json.dump(json_data, f, indent=2, default=str)
            
            result = {
                'status': 'success',
                'output_file': str(output_path),
                'data_size': len(json_data) if isinstance(json_data, list) else sum(len(v) for v in json_data.values()) if isinstance(json_data, dict) else 0,
                'schema_validated': self.schema_validation and schema is not None,
                'compact_format': self.compact_format,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"JSON export complete: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise ExportError(f"JSON export failed: {e}")
    
    def _dataframe_to_json(self, df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """Convert DataFrame to JSON-serializable format."""
        if df.empty:
            return []
        
        # Handle different data types
        json_records = []
        
        for idx, row in df.iterrows():
            record = {}
            
            # Add index as ID if it has a meaningful name
            if df.index.name and df.index.name != 'index':
                record[df.index.name] = self._serialize_value(idx)
            
            # Process each column
            for col in df.columns:
                value = row[col]
                record[col] = self._serialize_value(value)
            
            json_records.append(record)
        
        return json_records
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value to JSON-compatible format."""
        if pd.isna(value) or value is None:
            return None
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        elif isinstance(value, (dict, list, str, int, float, bool)):
            return value
        else:
            return str(value)
    
    def _add_metadata(self, json_data: Any, original_data: Any, **kwargs) -> Dict[str, Any]:
        """Add metadata to JSON data."""
        metadata = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "exporter": "multiomics-pipeline",
                "data_type": type(original_data).__name__,
                "schema_validation": self.schema_validation,
                "compact_format": self.compact_format
            },
            "data": json_data
        }
        
        # Add data-specific metadata
        if isinstance(original_data, pd.DataFrame):
            metadata["metadata"].update({
                "shape": original_data.shape,
                "columns": list(original_data.columns),
                "dtypes": original_data.dtypes.astype(str).to_dict(),
                "memory_usage": original_data.memory_usage(deep=True).sum()
            })
        elif isinstance(original_data, dict):
            metadata["metadata"]["datasets"] = {
                key: {
                    "shape": df.shape if hasattr(df, 'shape') else None,
                    "columns": list(df.columns) if hasattr(df, 'columns') else None
                }
                for key, df in original_data.items()
            }
        
        return metadata
    
    def _validate_json_schema(self, json_data: Any, schema: Dict[str, Any]):
        """Validate JSON data against schema."""
        try:
            import jsonschema
            
            # If data has metadata wrapper, validate the data part
            if isinstance(json_data, dict) and 'data' in json_data:
                data_to_validate = json_data['data']
            else:
                data_to_validate = json_data
            
            jsonschema.validate(instance=data_to_validate, schema=schema)
            logger.info("JSON schema validation passed")
            
        except ImportError:
            logger.warning("jsonschema library not available, skipping validation")
        except jsonschema.exceptions.ValidationError as e:
            raise ExportError(f"JSON schema validation failed: {e.message}")
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
    
    def export_to_jsonl(self, data: pd.DataFrame, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Export DataFrame to JSONL (JSON Lines) format.
        
        Args:
            data: Input DataFrame
            output_file: Output file path
            **kwargs: Additional parameters
            
        Returns:
            Export results
        """
        logger.info(f"Exporting data to JSONL format: {output_file}")
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            records_written = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, row in data.iterrows():
                    record = {}
                    
                    # Add index if meaningful
                    if data.index.name and data.index.name != 'index':
                        record[data.index.name] = self._serialize_value(idx)
                    
                    # Process columns
                    for col in data.columns:
                        record[col] = self._serialize_value(row[col])
                    
                    # Write as JSON line
                    json_line = json.dumps(record, default=str)
                    f.write(json_line + '\n')
                    records_written += 1
            
            result = {
                'status': 'success',
                'output_file': str(output_path),
                'records_written': records_written,
                'format': 'jsonl',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"JSONL export complete: {records_written} records")
            
            return result
            
        except Exception as e:
            logger.error(f"JSONL export failed: {e}")
            raise ExportError(f"JSONL export failed: {e}")
    
    def create_json_schema(self, data: pd.DataFrame, schema_name: str = "multiomics_schema") -> Dict[str, Any]:
        """
        Create JSON schema from DataFrame structure.
        
        Args:
            data: Input DataFrame
            schema_name: Name for the schema
            
        Returns:
            JSON schema dictionary
        """
        logger.info(f"Creating JSON schema: {schema_name}")
        
        # Analyze data types
        properties = {}
        required = []
        
        for col in data.columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                # All values are null
                properties[col] = {"type": ["null"]}
                continue
            
            # Determine JSON schema type
            if pd.api.types.is_integer_dtype(col_data):
                properties[col] = {
                    "type": ["integer", "null"],
                    "description": f"Integer column {col}"
                }
            elif pd.api.types.is_float_dtype(col_data):
                properties[col] = {
                    "type": ["number", "null"],
                    "description": f"Numeric column {col}"
                }
            elif pd.api.types.is_bool_dtype(col_data):
                properties[col] = {
                    "type": ["boolean", "null"],
                    "description": f"Boolean column {col}"
                }
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                properties[col] = {
                    "type": ["string", "null"],
                    "format": "date-time",
                    "description": f"DateTime column {col}"
                }
            else:
                # String type
                properties[col] = {
                    "type": ["string", "null"],
                    "description": f"String column {col}"
                }
            
            # Check if column has any null values
            if data[col].isnull().any():
                properties[col]["type"] = ["null"] + (properties[col]["type"] if isinstance(properties[col]["type"], list) else [properties[col]["type"]])
            else:
                required.append(col)
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": f"http://example.com/{schema_name}.json",
            "title": schema_name,
            "description": f"JSON schema for {schema_name}",
            "type": "array",
            "items": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
        return schema
    
    def export_feature_matrix(self, data: Dict[str, pd.DataFrame], output_file: str, 
                             sample_id_column: str = 'sample_id', **kwargs) -> Dict[str, Any]:
        """
        Export feature matrix to JSON format.
        
        Args:
            data: Dictionary of omics DataFrames
            output_file: Output file path
            sample_id_column: Column name for sample IDs
            **kwargs: Additional parameters
            
        Returns:
            Export results
        """
        logger.info("Exporting feature matrix to JSON")
        
        try:
            # Align data by sample ID
            aligned_data = self._align_data_by_samples(data, sample_id_column)
            
            # Create feature matrix
            feature_matrix = []
            sample_ids = None
            
            # Get all sample IDs
            for df in aligned_data.values():
                if sample_id_column in df.columns:
                    current_samples = set(df[sample_id_column].values)
                else:
                    current_samples = set(df.index.values)
                
                if sample_ids is None:
                    sample_ids = current_samples
                else:
                    sample_ids = sample_ids.intersection(current_samples)
            
            sample_ids = sorted(list(sample_ids))
            
            # Create feature matrix
            for sample_id in sample_ids:
                sample_data = {
                    sample_id_column: sample_id,
                    "features": {}
                }
                
                for omics_type, df in aligned_data.items():
                    if sample_id_column in df.columns:
                        sample_row = df[df[sample_id_column] == sample_id]
                    else:
                        sample_row = df.loc[sample_id:sample_id]
                    
                    if not sample_row.empty:
                        # Extract features (numeric columns only)
                        numeric_data = sample_row.select_dtypes(include=[np.number])
                        if not numeric_data.empty:
                            features = numeric_data.iloc[0].to_dict()
                            # Add prefix to avoid conflicts
                            prefixed_features = {f"{omics_type}_{k}": v for k, v in features.items()}
                            sample_data["features"].update(prefixed_features)
                
                feature_matrix.append(sample_data)
            
            # Export to JSON
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(feature_matrix, f, indent=2, default=str)
            
            result = {
                'status': 'success',
                'output_file': str(output_path),
                'n_samples': len(feature_matrix),
                'n_features': len(feature_matrix[0]["features"]) if feature_matrix else 0,
                'format': 'feature_matrix',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Feature matrix export complete: {result['n_samples']} samples, {result['n_features']} features")
            
            return result
            
        except Exception as e:
            logger.error(f"Feature matrix export failed: {e}")
            raise ExportError(f"Feature matrix export failed: {e}")
    
    def _align_data_by_samples(self, data: Dict[str, pd.DataFrame], 
                              sample_id_column: str) -> Dict[str, pd.DataFrame]:
        """Align data by sample IDs."""
        aligned_data = {}
        
        for omics_type, df in data.items():
            if df.empty:
                continue
            
            # Ensure sample ID column exists
            if sample_id_column not in df.columns:
                if df.index.name == sample_id_column:
                    df = df.reset_index()
                else:
                    logger.warning(f"Sample ID column '{sample_id_column}' not found in {omics_type}")
                    continue
            
            aligned_data[omics_type] = df
        
        return aligned_data
    
    def create_summary_report(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> str:
        """Create a summary report of the data."""
        report = []
        
        report.append("JSON EXPORT DATA SUMMARY")
        report.append("=" * 40)
        
        if isinstance(data, pd.DataFrame):
            report.append(f"Data Type: Single DataFrame")
            report.append(f"Shape: {data.shape}")
            report.append(f"Columns: {list(data.columns)}")
            report.append(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Data types summary
            dtype_counts = data.dtypes.value_counts()
            report.append(f"Data Types:")
            for dtype, count in dtype_counts.items():
                report.append(f"  {dtype}: {count} columns")
            
            # Missing data summary
            missing_pct = (data.isnull().sum() / len(data) * 100).round(2)
            high_missing = missing_pct[missing_pct > 50]
            if len(high_missing) > 0:
                report.append(f"Columns with >50% missing data: {list(high_missing.index)}")
        
        elif isinstance(data, dict):
            report.append(f"Data Type: Dictionary of DataFrames")
            report.append(f"Number of datasets: {len(data)}")
            
            for key, df in data.items():
                report.append(f"\nDataset: {key}")
                report.append(f"  Shape: {df.shape}")
                report.append(f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        
        return "\n".join(report)