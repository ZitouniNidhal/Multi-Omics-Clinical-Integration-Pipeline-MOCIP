"""Data validation utilities for multi-omics pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import logging
from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate multi-omics data using Pandera schemas."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.validation_level = self.config.get('validation_level', 'strict')
        self.allow_missing = self.config.get('allow_missing', False)
        
        # Predefined schemas
        self.schemas = self._initialize_schemas()
        
        logger.info(f"Initialized DataValidator with level: {self.validation_level}")
    
    def _initialize_schemas(self) -> Dict[str, DataFrameSchema]:
        """Initialize predefined validation schemas."""
        schemas = {}
        
        # Gene expression schema
        schemas['gene_expression'] = DataFrameSchema({
            "sample_id": Column(str, nullable=False),
            "gene_id": Column(str, nullable=False),
            "expression_value": Column(float, nullable=False, checks=Check.greater_than_or_equal_to(0)),
            "tpm": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "fpkm": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "gene_symbol": Column(str, nullable=True),
            "chromosome": Column(str, nullable=True),
            "start_position": Column(int, nullable=True),
            "end_position": Column(int, nullable=True),
            "strand": Column(str, nullable=True, checks=Check.isin(['+', '-', '.'])),
        })
        
        # Clinical data schema
        schemas['clinical'] = DataFrameSchema({
            "patient_id": Column(str, nullable=False),
            "sample_id": Column(str, nullable=False),
            "age": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "gender": Column(str, nullable=True, checks=Check.isin(['male', 'female', 'M', 'F', 'other'])),
            "tumor_stage": Column(str, nullable=True),
            "survival_time": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "vital_status": Column(str, nullable=True, checks=Check.isin(['alive', 'dead', 'deceased'])),
            "tumor_grade": Column(str, nullable=True),
            "tumor_size": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "lymph_nodes": Column(int, nullable=True, checks=Check.greater_than_or_equal_to(0)),
        })
        
        # Proteomics schema
        schemas['proteomics'] = DataFrameSchema({
            "sample_id": Column(str, nullable=False),
            "protein_id": Column(str, nullable=False),
            "protein_name": Column(str, nullable=True),
            "intensity": Column(float, nullable=False, checks=Check.greater_than_or_equal_to(0)),
            "normalized_intensity": Column(float, nullable=True),
            "gene_symbol": Column(str, nullable=True),
            "peptide_count": Column(int, nullable=True, checks=Check.greater_than_or_equal_to(1)),
        })
        
        # Metabolomics schema
        schemas['metabolomics'] = DataFrameSchema({
            "sample_id": Column(str, nullable=False),
            "metabolite_id": Column(str, nullable=False),
            "metabolite_name": Column(str, nullable=True),
            "concentration": Column(float, nullable=False, checks=Check.greater_than_or_equal_to(0)),
            "normalized_concentration": Column(float, nullable=True),
            "mz": Column(float, nullable=True, checks=Check.greater_than(0)),
            "retention_time": Column(float, nullable=True, checks=Check.greater_than(0)),
        })
        
        return schemas
    
    def validate_dataframe(self, df: pd.DataFrame, schema_name: str, 
                          custom_checks: Optional[Dict[str, List[Check]]] = None) -> Dict[str, Any]:
        """
        Validate DataFrame against predefined schema.
        
        Args:
            df: DataFrame to validate
            schema_name: Name of schema to use
            custom_checks: Additional custom checks
            
        Returns:
            Validation results
        """
        logger.info(f"Validating DataFrame against schema: {schema_name}")
        
        if schema_name not in self.schemas:
            raise ValidationError(f"Unknown schema: {schema_name}")
        
        schema = self.schemas[schema_name]
        
        # Add custom checks if provided
        if custom_checks:
            for col_name, checks in custom_checks.items():
                if col_name in schema.columns:
                    existing_checks = list(schema.columns[col_name].checks) if schema.columns[col_name].checks else []
                    existing_checks.extend(checks)
                    schema.columns[col_name].checks = existing_checks
        
        try:
            # Validate
            validated_df = schema.validate(df, lazy=True)
            
            results = {
                'valid': True,
                'schema': schema_name,
                'n_errors': 0,
                'errors': [],
                'warnings': [],
                'statistics': self._calculate_validation_statistics(df, schema)
            }
            
        except pa.errors.SchemaErrors as e:
            results = {
                'valid': False,
                'schema': schema_name,
                'n_errors': len(e.failure_cases),
                'errors': e.failure_cases.to_dict('records'),
                'warnings': [],
                'statistics': self._calculate_validation_statistics(df, schema)
            }
            
            # Log errors
            for error in results['errors']:
                logger.warning(f"Validation error: {error}")
        
        return results
    
    def validate_multi_omics_data(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate multiple omics datasets.
        
        Args:
            omics_data: Dictionary of omics DataFrames
            
        Returns:
            Validation results for each dataset
        """
        logger.info("Validating multi-omics data")
        
        results = {
            'overall_valid': True,
            'data_type_results': {},
            'summary': {
                'total_data_types': len(omics_data),
                'valid_data_types': 0,
                'invalid_data_types': 0,
                'total_errors': 0
            }
        }
        
        for data_type, df in omics_data.items():
            if df.empty:
                results['data_type_results'][data_type] = {
                    'valid': False,
                    'error': 'Empty DataFrame',
                    'warnings': ['No data available']
                }
                results['overall_valid'] = False
                results['summary']['invalid_data_types'] += 1
                continue
            
            # Determine appropriate schema
            schema_name = self._determine_schema(data_type, df)
            
            if schema_name:
                validation_result = self.validate_dataframe(df, schema_name)
                results['data_type_results'][data_type] = validation_result
                
                if not validation_result['valid']:
                    results['overall_valid'] = False
                    results['summary']['invalid_data_types'] += 1
                    results['summary']['total_errors'] += validation_result['n_errors']
                else:
                    results['summary']['valid_data_types'] += 1
            else:
                results['data_type_results'][data_type] = {
                    'valid': True,
                    'warnings': ['No specific schema available, using basic validation'],
                    'basic_validation': self._basic_validation(df)
                }
                results['summary']['valid_data_types'] += 1
        
        return results
    
    def _determine_schema(self, data_type: str, df: pd.DataFrame) -> Optional[str]:
        """Determine appropriate schema for data type."""
        data_type_lower = data_type.lower()
        
        # Direct mapping
        if 'gene_expression' in data_type_lower or 'transcriptomics' in data_type_lower:
            return 'gene_expression'
        elif 'clinical' in data_type_lower or 'patient' in data_type_lower:
            return 'clinical'
        elif 'proteomics' in data_type_lower or 'protein' in data_type_lower:
            return 'proteomics'
        elif 'metabolomics' in data_type_lower or 'metabolite' in data_type_lower:
            return 'metabolomics'
        
        # Column-based inference
        columns = set(df.columns.str.lower())
        
        if 'gene_id' in columns or 'ensembl_id' in columns:
            return 'gene_expression'
        elif 'patient_id' in columns or 'age' in columns or 'gender' in columns:
            return 'clinical'
        elif 'protein_id' in columns or 'intensity' in columns:
            return 'proteomics'
        elif 'metabolite_id' in columns or 'concentration' in columns:
            return 'metabolomics'
        
        return None
    
    def _basic_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic validation when no specific schema is available."""
        results = {
            'has_data': not df.empty,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'column_types': df.dtypes.to_dict(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return results
    
    def _calculate_validation_statistics(self, df: pd.DataFrame, 
                                       schema: DataFrameSchema) -> Dict[str, Any]:
        """Calculate validation statistics."""
        stats = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'column_coverage': len(set(df.columns) & set(schema.columns)) / len(schema.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            'data_types_match': {},
            'completeness_score': 0.0
        }
        
        # Check data type matches
        for col_name, col_schema in schema.columns.items():
            if col_name in df.columns:
                expected_dtype = col_schema.dtype
                actual_dtype = str(df[col_name].dtype)
                
                # Simple type checking
                type_match = self._check_dtype_match(expected_dtype, actual_dtype)
                stats['data_types_match'][col_name] = type_match
        
        # Calculate completeness score (0-100)
        type_matches = sum(stats['data_types_match'].values())
        total_schema_cols = len(schema.columns)
        
        stats['completeness_score'] = (
            (stats['column_coverage'] * 0.4 +  # Column presence
             (type_matches / total_schema_cols) * 0.4 +  # Type matching
             (1 - stats['missing_data_percentage'] / 100) * 0.2) * 100  # Data completeness
        )
        
        return stats
    
    def _check_dtype_match(self, expected: str, actual: str) -> bool:
        """Check if data types match."""
        # Simplified type checking
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        
        if 'int' in expected_lower:
            return 'int' in actual_lower
        elif 'float' in expected_lower:
            return 'float' in actual_lower or 'int' in actual_lower
        elif 'str' in expected_lower or 'object' in expected_lower:
            return 'object' in actual_lower or 'str' in actual_lower
        elif 'bool' in expected_lower:
            return 'bool' in actual_lower
        else:
            return expected_lower in actual_lower
    
    def validate_sample_ids(self, sample_ids: List[str], 
                          expected_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate sample IDs format and uniqueness.
        
        Args:
            sample_ids: List of sample IDs
            expected_format: Expected format pattern
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'unique_count': len(set(sample_ids)),
            'total_count': len(sample_ids),
            'duplicates': [],
            'format_errors': [],
            'format_validation': {}
        }
        
        # Check uniqueness
        if len(set(sample_ids)) != len(sample_ids):
            duplicates = [item for item, count in pd.Series(sample_ids).value_counts().items() if count > 1]
            results['duplicates'] = duplicates
            results['valid'] = False
        
        # Check format if expected format provided
        if expected_format:
            import re
            
            for sample_id in sample_ids:
                if not re.match(expected_format, sample_id):
                    results['format_errors'].append(sample_id)
                    results['valid'] = False
            
            results['format_validation'] = {
                'expected_format': expected_format,
                'valid_format_count': len(sample_ids) - len(results['format_errors']),
                'invalid_format_count': len(results['format_errors'])
            }
        
        return results
    
    def validate_numeric_ranges(self, df: pd.DataFrame, 
                              column_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Validate numeric columns are within specified ranges.
        
        Args:
            df: DataFrame to validate
            column_ranges: Dictionary of column names to (min, max) tuples
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'column_results': {},
            'out_of_range_values': {}
        }
        
        for col_name, (min_val, max_val) in column_ranges.items():
            if col_name in df.columns:
                col_data = df[col_name].dropna()
                
                if pd.api.types.is_numeric_dtype(col_data):
                    out_of_range = col_data[(col_data < min_val) | (col_data > max_val)]
                    
                    results['column_results'][col_name] = {
                        'min_expected': min_val,
                        'max_expected': max_val,
                        'min_actual': col_data.min(),
                        'max_actual': col_data.max(),
                        'out_of_range_count': len(out_of_range),
                        'valid': len(out_of_range) == 0
                    }
                    
                    if len(out_of_range) > 0:
                        results['valid'] = False
                        results['out_of_range_values'][col_name] = out_of_range.tolist()
                else:
                    results['column_results'][col_name] = {
                        'error': 'Column is not numeric',
                        'valid': False
                    }
                    results['valid'] = False
        
        return results
    
    def create_validation_report(self, validation_results: Dict[str, Any], 
                               output_path: Optional[str] = None) -> str:
        """Create a detailed validation report."""
        report_lines = []
        
        report_lines.append("MULTI-OMICS DATA VALIDATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Validation Timestamp: {pd.Timestamp.now().isoformat()}")
        report_lines.append(f"Overall Validation: {'PASSED' if validation_results.get('overall_valid', False) else 'FAILED'}")
        report_lines.append("")
        
        # Summary statistics
        if 'summary' in validation_results:
            summary = validation_results['summary']
            report_lines.append("SUMMARY STATISTICS:")
            report_lines.append(f"  Total Data Types: {summary.get('total_data_types', 0)}")
            report_lines.append(f"  Valid Data Types: {summary.get('valid_data_types', 0)}")
            report_lines.append(f"  Invalid Data Types: {summary.get('invalid_data_types', 0)}")
            report_lines.append(f"  Total Errors: {summary.get('total_errors', 0)}")
            report_lines.append("")
        
        # Detailed results for each data type
        if 'data_type_results' in validation_results:
            report_lines.append("DETAILED VALIDATION RESULTS:")
            
            for data_type, results in validation_results['data_type_results'].items():
                report_lines.append(f"\n{data_type.upper()}:")
                
                if isinstance(results, dict):
                    if 'valid' in results:
                        report_lines.append(f"  Validation Status: {'PASSED' if results['valid'] else 'FAILED'}")
                    
                    if 'n_errors' in results:
                        report_lines.append(f"  Number of Errors: {results['n_errors']}")
                    
                    if 'errors' in results and results['errors']:
                        report_lines.append("  Errors:")
                        for error in results['errors'][:5]:  # Show first 5 errors
                            report_lines.append(f"    - {error}")
                        if len(results['errors']) > 5:
                            report_lines.append(f"    ... and {len(results['errors']) - 5} more errors")
                    
                    if 'warnings' in results and results['warnings']:
                        report_lines.append("  Warnings:")
                        for warning in results['warnings']:
                            report_lines.append(f"    - {warning}")
                    
                    if 'statistics' in results:
                        stats = results['statistics']
                        report_lines.append("  Statistics:")
                        report_lines.append(f"    Rows: {stats.get('n_rows', 'N/A')}")
                        report_lines.append(f"    Columns: {stats.get('n_columns', 'N/A')}")
                        report_lines.append(f"    Completeness Score: {stats.get('completeness_score', 0):.1f}%")
                        report_lines.append(f"    Missing Data: {stats.get('missing_data_percentage', 0):.1f}%")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Validation report saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save validation report: {e}")
        
        return report_text


class SchemaBuilder:
    """Build custom validation schemas."""
    
    @staticmethod
    def build_gene_expression_schema(required_columns: Optional[List[str]] = None) -> DataFrameSchema:
        """Build gene expression validation schema."""
        base_schema = {
            "sample_id": Column(str, nullable=False),
            "gene_id": Column(str, nullable=False),
            "expression_value": Column(float, nullable=False, checks=Check.greater_than_or_equal_to(0))
        }
        
        optional_columns = {
            "gene_symbol": Column(str, nullable=True),
            "tpm": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "fpkm": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "chromosome": Column(str, nullable=True),
            "start_position": Column(int, nullable=True),
            "end_position": Column(int, nullable=True),
            "strand": Column(str, nullable=True, checks=Check.isin(['+', '-', '.']))
        }
        
        # Add required optional columns
        if required_columns:
            for col in required_columns:
                if col in optional_columns:
                    base_schema[col] = optional_columns[col]
        
        return DataFrameSchema(base_schema)
    
    @staticmethod
    def build_clinical_schema(required_columns: Optional[List[str]] = None) -> DataFrameSchema:
        """Build clinical data validation schema."""
        base_schema = {
            "patient_id": Column(str, nullable=False),
            "sample_id": Column(str, nullable=False)
        }
        
        optional_columns = {
            "age": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "gender": Column(str, nullable=True, checks=Check.isin(['male', 'female', 'M', 'F', 'other'])),
            "tumor_stage": Column(str, nullable=True),
            "survival_time": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "vital_status": Column(str, nullable=True, checks=Check.isin(['alive', 'dead', 'deceased'])),
            "tumor_grade": Column(str, nullable=True),
            "tumor_size": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "lymph_nodes": Column(int, nullable=True, checks=Check.greater_than_or_equal_to(0)),
            "ethnicity": Column(str, nullable=True),
            "race": Column(str, nullable=True)
        }
        
        # Add required optional columns
        if required_columns:
            for col in required_columns:
                if col in optional_columns:
                    base_schema[col] = optional_columns[col]
        
        return DataFrameSchema(base_schema)