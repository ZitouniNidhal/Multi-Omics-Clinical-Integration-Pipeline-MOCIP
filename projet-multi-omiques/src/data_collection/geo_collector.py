"""GEO data collector using GEOparse and direct API access."""

import pandas as pd
import GEOparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from .base_collector import BaseCollector
from ..exceptions import GEOError, DataCollectionError

logger = logging.getLogger(__name__)


class GEOCollector(BaseCollector):
    """Collector for GEO data using GEOparse library."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GEO collector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config.get('geo', {}), config.get('geo', {}).get('download_dir'))
        
        self.ftp_endpoint = config.get('geo', {}).get('ftp_endpoint', 'ftp.ncbi.nlm.nih.gov')
        self.search_limit = config.get('geo', {}).get('search_limit', 100)
        self.datasets = config.get('geo', {}).get('datasets', [])
        
        logger.info(f"Initialized GEO collector with endpoint: {self.ftp_endpoint}")
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for GEO datasets.
        
        Args:
            query: Search query (e.g., "breast cancer", "GSE96058")
            **kwargs: Additional search parameters
            
        Returns:
            List of matching datasets
        """
        try:
            # Use GEOparse search functionality
            search_results = GEOparse.search(query, **kwargs)
            
            datasets = []
            for gse in search_results[:self.search_limit]:
                datasets.append({
                    'id': gse.get('accession', ''),
                    'title': gse.get('title', ''),
                    'summary': gse.get('summary', ''),
                    'organism': gse.get('organism', ''),
                    'platform': gse.get('platform', ''),
                    'samples': gse.get('samples', 0),
                    'submission_date': gse.get('submission_date', ''),
                    'source': 'GEO'
                })
            
            logger.info(f"Found {len(datasets)} GEO datasets for query: {query}")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to search GEO datasets: {e}")
            raise GEOError(f"GEO search failed: {e}")
    
    def validate_dataset(self, dataset_id: str) -> bool:
        """
        Validate if a GEO dataset exists.
        
        Args:
            dataset_id: GEO dataset ID (e.g., GSE96058)
            
        Returns:
            True if dataset exists
        """
        try:
            # Try to get dataset information
            gse = GEOparse.get_GEO(dataset_id, destdir=self.download_dir, silent=True)
            
            if gse is not None:
                logger.info(f"Validated GEO dataset: {dataset_id}")
                return True
            else:
                logger.warning(f"GEO dataset not found: {dataset_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate GEO dataset {dataset_id}: {e}")
            return False
    
    def collect(self, dataset_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Collect GEO dataset.
        
        Args:
            dataset_id: GEO dataset ID (e.g., GSE96058)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing collected data
        """
        if not dataset_id:
            dataset_id = kwargs.get('gse_id', self.datasets[0] if self.datasets else None)
        
        if not dataset_id:
            raise GEOError("No GEO dataset ID specified")
        
        logger.info(f"Collecting GEO dataset: {dataset_id}")
        
        # Validate dataset
        if not self.validate_dataset(dataset_id):
            raise GEOError(f"Invalid GEO dataset: {dataset_id}")
        
        try:
            # Download and parse GEO dataset
            gse = GEOparse.get_GEO(dataset_id, destdir=self.download_dir, silent=False)
            
            collected_data = {
                'dataset_id': dataset_id,
                'collection_timestamp': datetime.now().isoformat(),
                'data': {}
            }
            
            # Extract platform information
            platforms = {}
            for platform_id, platform in gse.gpls.items():
                platforms[platform_id] = {
                    'name': platform.name,
                    'technology': platform.technology,
                    'organism': platform.organism,
                    'num_probes': len(platform.table) if hasattr(platform, 'table') else 0
                }
            
            collected_data['platforms'] = platforms
            
            # Collect different types of data
            # 1. Gene expression data
            if hasattr(gse, 'pivot_samples') and len(gse.pivot_samples) > 0:
                expression_data = self._extract_expression_data(gse)
                collected_data['data']['gene_expression'] = expression_data
            
            # 2. Sample metadata (phenotype data)
            sample_metadata = self._extract_sample_metadata(gse)
            collected_data['data']['sample_metadata'] = sample_metadata
            
            # 3. Platform annotation
            platform_annotation = self._extract_platform_annotation(gse)
            collected_data['data']['platform_annotation'] = platform_annotation
            
            # Save metadata
            self.save_metadata(collected_data, f"geo_{dataset_id.lower()}")
            
            logger.info(f"Successfully collected GEO dataset: {dataset_id}")
            return collected_data
            
        except Exception as e:
            logger.error(f"Failed to collect GEO dataset {dataset_id}: {e}")
            raise GEOError(f"GEO collection failed: {e}")
    
    def _extract_expression_data(self, gse) -> Dict[str, Any]:
        """Extract gene expression data from GEO dataset."""
        try:
            logger.info("Extracting gene expression data")
            
            # Get expression matrix
            if hasattr(gse, 'pivot_samples'):
                expression_matrix = gse.pivot_samples('VALUE')
                
                if expression_matrix is not None and not expression_matrix.empty:
                    # Transpose to have genes as rows, samples as columns
                    expression_matrix = expression_matrix.T
                    
                    logger.info(f"Expression matrix shape: {expression_matrix.shape}")
                    
                    return {
                        'expression_matrix': expression_matrix,
                        'processing_info': {
                            'num_genes': expression_matrix.shape[0],
                            'num_samples': expression_matrix.shape[1],
                            'missing_values': expression_matrix.isnull().sum().sum(),
                            'data_type': 'continuous'
                        }
                    }
            
            # Alternative: extract from individual samples
            expression_data = {}
            sample_info = []
            
            for sample_name, sample in gse.gsms.items():
                if hasattr(sample, 'table') and not sample.table.empty:
                    # Assume first column is probe ID and second is expression value
                    probe_col = sample.table.columns[0]
                    value_col = sample.table.columns[1] if len(sample.table.columns) > 1 else probe_col
                    
                    sample_expr = sample.table.set_index(probe_col)[value_col]
                    expression_data[sample_name] = sample_expr
                    
                    sample_info.append({
                        'sample_id': sample_name,
                        'source_name': sample.metadata.get('source_name_ch1', [''])[0] if 'source_name_ch1' in sample.metadata else '',
                        'organism': sample.metadata.get('organism_ch1', [''])[0] if 'organism_ch1' in sample.metadata else '',
                        'characteristics': self._parse_characteristics(sample.metadata)
                    })
            
            if expression_data:
                expression_matrix = pd.DataFrame(expression_data)
                
                return {
                    'expression_matrix': expression_matrix,
                    'sample_info': pd.DataFrame(sample_info),
                    'processing_info': {
                        'num_genes': expression_matrix.shape[0],
                        'num_samples': expression_matrix.shape[1],
                        'missing_values': expression_matrix.isnull().sum().sum(),
                        'data_type': 'continuous'
                    }
                }
            
            logger.warning("No expression data found")
            return {
                'expression_matrix': pd.DataFrame(),
                'processing_info': {'num_genes': 0, 'num_samples': 0}
            }
            
        except Exception as e:
            logger.error(f"Failed to extract expression data: {e}")
            raise GEOError(f"Expression data extraction failed: {e}")
    
    def _extract_sample_metadata(self, gse) -> Dict[str, Any]:
        """Extract sample metadata from GEO dataset."""
        try:
            logger.info("Extracting sample metadata")
            
            sample_metadata = []
            
            for sample_name, sample in gse.gsms.items():
                metadata = {
                    'sample_id': sample_name,
                    'title': sample.metadata.get('title', [''])[0] if 'title' in sample.metadata else '',
                    'source_name': sample.metadata.get('source_name_ch1', [''])[0] if 'source_name_ch1' in sample.metadata else '',
                    'organism': sample.metadata.get('organism_ch1', [''])[0] if 'organism_ch1' in sample.metadata else '',
                    'characteristics': self._parse_characteristics(sample.metadata),
                    'treatment_protocol': sample.metadata.get('treatment_protocol_ch1', [''])[0] if 'treatment_protocol_ch1' in sample.metadata else '',
                    'growth_protocol': sample.metadata.get('growth_protocol_ch1', [''])[0] if 'growth_protocol_ch1' in sample.metadata else '',
                    'molecule': sample.metadata.get('molecule_ch1', [''])[0] if 'molecule_ch1' in sample.metadata else '',
                    'platform_id': list(sample.metadata.get('platform_id', [])),
                }
                
                sample_metadata.append(metadata)
            
            sample_df = pd.DataFrame(sample_metadata)
            
            logger.info(f"Extracted metadata for {len(sample_df)} samples")
            
            return {
                'sample_metadata': sample_df,
                'processing_info': {
                    'num_samples': len(sample_df),
                    'organisms': sample_df['organism'].unique().tolist() if 'organism' in sample_df.columns else [],
                    'platforms': sample_df['platform_id'].explode().unique().tolist() if 'platform_id' in sample_df.columns else []
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to extract sample metadata: {e}")
            raise GEOError(f"Sample metadata extraction failed: {e}")
    
    def _extract_platform_annotation(self, gse) -> Dict[str, Any]:
        """Extract platform annotation from GEO dataset."""
        try:
            logger.info("Extracting platform annotation")
            
            platform_annotations = {}
            
            for platform_id, platform in gse.gpls.items():
                if hasattr(platform, 'table') and not platform.table.empty:
                    platform_annotations[platform_id] = {
                        'annotation_table': platform.table,
                        'platform_info': {
                            'name': platform.name,
                            'technology': platform.technology,
                            'organism': platform.organism,
                            'manufacturer': platform.metadata.get('manufacturer', [''])[0] if 'manufacturer' in platform.metadata else '',
                            'distribution': platform.metadata.get('distribution', [''])[0] if 'distribution' in platform.metadata else '',
                        }
                    }
            
            logger.info(f"Extracted annotation for {len(platform_annotations)} platforms")
            
            return {
                'platform_annotations': platform_annotations,
                'processing_info': {
                    'num_platforms': len(platform_annotations),
                    'platform_ids': list(platform_annotations.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to extract platform annotation: {e}")
            raise GEOError(f"Platform annotation extraction failed: {e}")
    
    def _parse_characteristics(self, metadata: Dict) -> Dict[str, str]:
        """Parse characteristics from sample metadata."""
        characteristics = {}
        
        # Look for characteristics_ch1 fields
        for key, values in metadata.items():
            if key.startswith('characteristics_ch1'):
                # Parse characteristics in format "key: value"
                for value in values:
                    if ':' in value:
                        char_key, char_value = value.split(':', 1)
                        characteristics[char_key.strip()] = char_value.strip()
        
        return characteristics
    
    def get_series_matrix(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Get series matrix file for a GEO dataset."""
        try:
            # Download series matrix file
            series_matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_id[:-3]}nnn/{dataset_id}/matrix/{dataset_id}_series_matrix.txt.gz"
            
            local_file = self.download_file(series_matrix_url)
            
            if local_file and local_file.exists():
                # Parse series matrix
                # This is a simplified implementation - real parsing would be more complex
                with open(local_file, 'r') as f:
                    lines = f.readlines()
                
                # Find data section
                data_start = None
                data_end = None
                
                for i, line in enumerate(lines):
                    if line.startswith('!series_matrix_table_begin'):
                        data_start = i + 1
                    elif line.startswith('!series_matrix_table_end'):
                        data_end = i
                        break
                
                if data_start and data_end:
                    # Parse data
                    data_lines = lines[data_start:data_end]
                    # This would need proper parsing logic
                    logger.info(f"Found series matrix with {len(data_lines)} data lines")
                    
                    # For now, return None to indicate we found it but need proper parsing
                    return None
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get series matrix for {dataset_id}: {e}")
            return None