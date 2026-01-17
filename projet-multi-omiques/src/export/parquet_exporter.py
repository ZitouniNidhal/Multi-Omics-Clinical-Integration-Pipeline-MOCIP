"""Parquet exporter for efficient multi-omics data storage."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import logging
from ..exceptions import ExportError

logger = logging.getLogger(__name__)


class ParquetExporter:
    """Export multi-omics data to Parquet format for efficient storage."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Parquet exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.compression = self.config.get('compression', 'snappy')
        self.partition_cols = self.config.get('partition_cols', [])
        self.row_group_size = self.config.get('row_group_size', 100000)
        self.engine = self.config.get('engine', 'pyarrow')
        
        logger.info(f"Initialized ParquetExporter with compression: {self.compression}")
    
    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Export data to Parquet format.
        
        Args:
            data: DataFrame or dictionary of DataFrames
            output_file: Output file path
            **kwargs: Additional parameters
            
        Returns:
            Export results
        """
        logger.info(f"Exporting data to Parquet format: {output_file}")
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                # Single DataFrame
                result = self._export_single_dataframe(data, output_path, **kwargs)
            elif isinstance(data, dict):
                # Multiple DataFrames - save as partitioned dataset
                result = self._export_multiple_dataframes(data, output_path, **kwargs)
            else:
                raise ExportError(f"Unsupported data type: {type(data)}")
            
            logger.info(f"Parquet export complete: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            raise ExportError(f"Parquet export failed: {e}")
    
    def _export_single_dataframe(self, df: pd.DataFrame, output_path: Path, 
                                **kwargs) -> Dict[str, Any]:
        """Export single DataFrame to Parquet."""
        if df.empty:
            raise ExportError("Cannot export empty DataFrame")
        
        # Optimize data types for Parquet
        optimized_df = self._optimize_dataframe_for_parquet(df)
        
        # Convert to Arrow Table
        table = pa.Table.from_pandas(optimized_df)
        
        # Configure Parquet writer
        write_options = {
            'compression': self.compression,
            'row_group_size': self.row_group_size,
            'version': '2.6'  # Use latest Parquet version
        }
        
        # Add partitioning if specified
        if self.partition_cols and all(col in df.columns for col in self.partition_cols):
            # Create partitioned dataset
            output_dir = output_path.parent / output_path.stem
            pq.write_to_dataset(
                table,
                root_path=output_dir,
                partition_cols=self.partition_cols,
                **write_options
            )
            actual_output = output_dir
        else:
            # Write single file
            pq.write_table(table, output_path, **write_options)
            actual_output = output_path
        
        # Get file statistics
        file_stats = self._get_parquet_stats(actual_output)
        
        result = {
            'status': 'success',
            'output_path': str(actual_output),
            'format': 'parquet',
            'compression': self.compression,
            'original_shape': df.shape,
            'optimized_shape': optimized_df.shape,
            'file_size_mb': file_stats['size_mb'],
            'compression_ratio': file_stats['compression_ratio'],
            'row_groups': file_stats['row_groups'],
            'partitioned': len(self.partition_cols) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _export_multiple_dataframes(self, data: Dict[str, pd.DataFrame], 
                                   output_path: Path, **kwargs) -> Dict[str, Any]:
        """Export multiple DataFrames as partitioned Parquet dataset."""
        output_dir = output_path.parent / output_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_rows = 0
        total_size_mb = 0
        dataset_stats = {}
        
        for name, df in data.items():
            if df.empty:
                logger.warning(f"Skipping empty DataFrame: {name}")
                continue
            
            # Optimize and convert to Arrow Table
            optimized_df = self._optimize_dataframe_for_parquet(df)
            table = pa.Table.from_pandas(optimized_df)
            
            # Write to separate file
            file_path = output_dir / f"{name}.parquet"
            
            pq.write_table(
                table,
                file_path,
                compression=self.compression,
                row_group_size=self.row_group_size
            )
            
            # Collect stats
            file_stats = self._get_parquet_stats(file_path)
            dataset_stats[name] = {
                'shape': df.shape,
                'file_size_mb': file_stats['size_mb'],
                'compression_ratio': file_stats['compression_ratio']
            }
            
            total_rows += len(df)
            total_size_mb += file_stats['size_mb']
        
        result = {
            'status': 'success',
            'output_directory': str(output_dir),
            'format': 'parquet_dataset',
            'compression': self.compression,
            'datasets': dataset_stats,
            'total_rows': total_rows,
            'total_size_mb': total_size_mb,
            'n_datasets': len(dataset_stats),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _optimize_dataframe_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for Parquet storage."""
        optimized_df = df.copy()
        
        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1:  # Less than 10% unique values
                optimized_df[col] = optimized_df[col].astype('category')
                logger.debug(f"Converted {col} to category (unique ratio: {unique_ratio:.2f})")
        
        # Optimize integer columns
        for col in df.select_dtypes(include=['int64', 'int32']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    optimized_df[col] = optimized_df[col].astype('uint8')
                elif col_max < 65535:
                    optimized_df[col] = optimized_df[col].astype('uint16')
                elif col_max < 4294967295:
                    optimized_df[col] = optimized_df[col].astype('uint32')
            else:  # Signed integers
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].notna().any():
                float_values = df[col].dropna()
                if np.all(np.abs(float_values) < 3.4e38):  # Within float32 range
                    optimized_df[col] = optimized_df[col].astype('float32')
                    logger.debug(f"Converted {col} to float32")
        
        return optimized_df
    
    def _get_parquet_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get Parquet file statistics."""
        try:
            if file_path.is_file():
                # Single file
                file_size = file_path.stat().st_size
                parquet_file = pq.ParquetFile(file_path)
                
                metadata = parquet_file.metadata
                row_groups = metadata.num_row_groups
                
                return {
                    'size_mb': file_size / (1024 * 1024),
                    'compression_ratio': None,  # Would need original size for calculation
                    'row_groups': row_groups,
                    'num_rows': metadata.num_rows,
                    'num_columns': metadata.num_columns,
                    'schema': str(metadata.schema)
                }
            elif file_path.is_dir():
                # Directory (partitioned dataset)
                total_size = sum(f.stat().st_size for f in file_path.rglob('*.parquet'))
                return {
                    'size_mb': total_size / (1024 * 1024),
                    'compression_ratio': None,
                    'row_groups': 'multiple',
                    'num_rows': 'varies',
                    'num_columns': 'varies',
                    'schema': 'partitioned'
                }
            else:
                return {
                    'size_mb': 0,
                    'compression_ratio': None,
                    'row_groups': 0,
                    'num_rows': 0,
                    'num_columns': 0,
                    'schema': 'unknown'
                }
        except Exception as e:
            logger.warning(f"Could not get Parquet stats for {file_path}: {e}")
            return {
                'size_mb': 0,
                'compression_ratio': None,
                'row_groups': 0,
                'num_rows': 0,
                'num_columns': 0,
                'schema': 'error'
            }
    
    def read_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read Parquet file into DataFrame.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional parameters for pandas.read_parquet
            
        Returns:
            DataFrame
        """
        logger.info(f"Reading Parquet file: {file_path}")
        
        try:
            # Use pandas to read Parquet
            df = pd.read_parquet(file_path, engine=self.engine, **kwargs)
            
            logger.info(f"Successfully read Parquet: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read Parquet file: {e}")
            raise ExportError(f"Failed to read Parquet file: {e}")
    
    def export_compressed_dataset(self, data: Dict[str, pd.DataFrame], 
                                 output_file: str, compression_levels: Dict[str, int] = None, **kwargs) -> Dict[str, Any]:
        """
        Export with different compression levels for comparison.
        
        Args:
            data: Dictionary of DataFrames
            output_file: Base output file path
            compression_levels: Compression levels to test
            **kwargs: Additional parameters
            
        Returns:
            Comparison results
        """
        if compression_levels is None:
            compression_levels = {
                'none': 0,
                'snappy': 1,
                'gzip': 9,
                'brotli': 11,
                'lz4': 1
            }
        
        logger.info("Testing different compression levels")
        
        results = {
            'compression_comparison': {},
            'original_size_mb': 0,
            'best_compression': None,
            'best_ratio': 0
        }
        
        # Calculate original size
        total_original_size = sum(df.memory_usage(deep=True).sum() for df in data.values())
        results['original_size_mb'] = total_original_size / (1024 * 1024)
        
        base_path = Path(output_file)
        
        for compression, level in compression_levels.items():
            try:
                # Create output path for this compression
                comp_output = base_path.parent / f"{base_path.stem}_{compression}{base_path.suffix}"
                
                # Export with this compression
                original_compression = self.compression
                self.compression = compression
                
                result = self.export(data, str(comp_output), **kwargs)
                
                # Restore original compression
                self.compression = original_compression
                
                # Store results
                comp_stats = {
                    'compression': compression,
                    'level': level,
                    'file_size_mb': result['file_size_mb'],
                    'compression_ratio': results['original_size_mb'] / result['file_size_mb'] if result['file_size_mb'] > 0 else 0,
                    'space_savings_pct': (1 - result['file_size_mb'] / results['original_size_mb']) * 100 if results['original_size_mb'] > 0 else 0
                }
                
                results['compression_comparison'][compression] = comp_stats
                
                # Track best compression
                if comp_stats['compression_ratio'] > results['best_ratio']:
                    results['best_ratio'] = comp_stats['compression_ratio']
                    results['best_compression'] = compression
                
            except Exception as e:
                logger.warning(f"Compression {compression} failed: {e}")
                results['compression_comparison'][compression] = {
                    'error': str(e),
                    'compression': compression
                }
        
        return results
    
    def create_parquet_summary(self, file_path: str) -> Dict[str, Any]:
        """
        Create summary of Parquet file contents.
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            Summary information
        """
        logger.info(f"Creating Parquet summary for: {file_path}")
        
        try:
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            schema = parquet_file.schema
            
            # Get column information
            column_info = []
            for i in range(metadata.num_columns):
                col_meta = metadata.row_group(0).column(i)
                column_info.append({
                    'name': schema.column(i).name,
                    'type': str(schema.column(i).type),
                    'compression': col_meta.compression,
                    'encoding': col_meta.encodings,
                    'null_count': col_meta.statistics.null_count,
                    'distinct_count': col_meta.statistics.distinct_count,
                    'min_value': col_meta.statistics.min,
                    'max_value': col_meta.statistics.max
                })
            
            summary = {
                'file_path': file_path,
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'num_row_groups': metadata.num_row_groups,
                'format_version': metadata.format_version,
                'created_by': metadata.created_by,
                'serialized_size': metadata.serialized_size,
                'columns': column_info,
                'compression_stats': self._get_compression_stats(parquet_file)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create Parquet summary: {e}")
            return {'error': str(e), 'file_path': file_path}
    
    def _get_compression_stats(self, parquet_file: pq.ParquetFile) -> Dict[str, Any]:
        """Get compression statistics for Parquet file."""
        try:
            metadata = parquet_file.metadata
            total_uncompressed = 0
            total_compressed = 0
            
            for rg_idx in range(metadata.num_row_groups):
                rg = metadata.row_group(rg_idx)
                total_uncompressed += rg.total_byte_size
                total_compressed += rg.total_compressed_size
            
            return {
                'total_uncompressed_bytes': total_uncompressed,
                'total_compressed_bytes': total_compressed,
                'overall_compression_ratio': total_uncompressed / total_compressed if total_compressed > 0 else 0,
                'space_savings_pct': (1 - total_compressed / total_uncompressed) * 100 if total_uncompressed > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Could not get compression stats: {e}")
            return {}
        