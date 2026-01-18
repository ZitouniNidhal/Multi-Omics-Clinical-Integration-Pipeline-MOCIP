"""Utility functions for data collection."""

import re
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from ..exceptions import DataCollectionError

logger = logging.getLogger(__name__)


def generate_sample_id(source: str, original_id: str, additional_info: Dict[str, Any] = None) -> str:
    """
    Generate a standardized sample ID.
    
    Args:
        source: Data source (e.g., 'TCGA', 'GEO')
        original_id: Original sample ID
        additional_info: Additional information for hashing
        
    Returns:
        Standardized sample ID
    """
    # Create a hash of the original ID and additional info
    hash_input = f"{source}:{original_id}"
    if additional_info:
        # Sort keys to ensure consistent hashing
        hash_input += ":" + str(sorted(additional_info.items()))
    
    hash_obj = hashlib.md5(hash_input.encode())
    hash_suffix = hash_obj.hexdigest()[:8]
    
    # Clean original ID
    clean_id = re.sub(r'[^A-Za-z0-9_-]', '_', original_id)
    
    # Create standardized ID
    standardized_id = f"{source.upper()}_{clean_id}_{hash_suffix}".upper()
    
    # Ensure uniqueness and reasonable length
    if len(standardized_id) > 50:
        standardized_id = standardized_id[:42] + hash_suffix
    
    return standardized_id


def map_ensembl_to_entrez(ensembl_ids: List[str], species: str = "human") -> Dict[str, str]:
    """
    Map Ensembl IDs to Entrez IDs.
    
    Args:
        ensembl_ids: List of Ensembl IDs
        species: Species name
        
    Returns:
        Dictionary mapping Ensembl to Entrez IDs
    """
    # This is a placeholder implementation
    # In practice, you would use biomart or similar service
    
    mapping = {}
    
    try:
        # Example using mygene.info API
        import mygene
        
        mg = mygene.MyGeneInfo()
        result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='entrezgene', species=species)
        
        for item in result:
            ensembl_id = item.get('query')
            entrez_id = item.get('entrezgene')
            if ensembl_id and entrez_id:
                mapping[ensembl_id] = str(entrez_id)
                
    except Exception as e:
        logger.warning(f"Failed to map Ensembl to Entrez IDs: {e}")
        
    return mapping


def download_files_parallel(urls: List[str], download_dir: Path, 
                          max_workers: int = 4, **kwargs) -> List[Path]:
    """
    Download multiple files in parallel.
    
    Args:
        urls: List of URLs to download
        download_dir: Directory to save files
        max_workers: Maximum number of parallel downloads
        **kwargs: Additional arguments for download
        
    Returns:
        List of downloaded file paths
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_single(url: str) -> Tuple[str, Optional[Path], Optional[str]]:
        try:
            # Get filename from URL or headers
            from urllib.parse import urlparse
            filename = urlparse(url).path.split('/')[-1]
            if not filename:
                filename = f"download_{hashlib.md5(url.encode()).hexdigest()[:8]}"
            
            file_path = download_dir / filename
            
            # Download with progress tracking
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (8192 * 50) == 0:  # Log every 50 chunks
                                logger.debug(f"Download progress for {filename}: {progress:.1f}%")
            
            return url, file_path, None
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return url, None, str(e)
    
    # Download files in parallel
    downloaded_files = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {executor.submit(download_single, url): url for url in urls}
        
        # Collect results
        for future in as_completed(future_to_url):
            url, file_path, error = future.result()
            
            if file_path:
                downloaded_files.append(file_path)
                logger.info(f"Successfully downloaded: {file_path.name}")
            else:
                errors.append(f"{url}: {error}")
    
    if errors:
        logger.warning(f"Download errors: {errors}")
    
    return downloaded_files


def parse_soft_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse SOFT format file.
    
    Args:
        file_path: Path to SOFT file
        
    Returns:
        Parsed data dictionary
    """
    try:
        data = {
            'header': {},
            'entities': []
        }
        
        current_entity = None
        current_section = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                # Header section
                if line.startswith('^'):
                    # New entity
                    if current_entity:
                        data['entities'].append(current_entity)
                    
                    parts = line[1:].split(' = ', 1)
                    entity_type = parts[0]
                    entity_id = parts[1] if len(parts) > 1 else ''
                    
                    current_entity = {
                        'type': entity_type,
                        'id': entity_id,
                        'attributes': {}
                    }
                
                elif line.startswith('!'):
                    # Attribute
                    if ' = ' in line:
                        key, value = line[1:].split(' = ', 1)
                        
                        if current_entity:
                            if key in current_entity['attributes']:
                                # Multiple values for same key
                                if not isinstance(current_entity['attributes'][key], list):
                                    current_entity['attributes'][key] = [current_entity['attributes'][key]]
                                current_entity['attributes'][key].append(value)
                            else:
                                current_entity['attributes'][key] = value
                        else:
                            # Header attribute
                            data['header'][key] = value
                
                elif line.startswith('#'):
                    # Comment or special directive
                    continue
                
                else:
                    # Data section
                    if current_entity and 'data' not in current_entity:
                        current_entity['data'] = []
                    
                    if current_entity:
                        current_entity['data'].append(line)
        
        # Add last entity
        if current_entity:
            data['entities'].append(current_entity)
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to parse SOFT file {file_path}: {e}")
        raise DataCollectionError(f"SOFT parsing failed: {e}")


def validate_sample_mapping(sample_ids_1: List[str], sample_ids_2: List[str], 
                            tolerance: float = 0.8) -> Dict[str, Any]:
    """
    Validate mapping between two sets of sample IDs.
    
    Args:
        sample_ids_1: First list of sample IDs
        sample_ids_2: Second list of sample IDs
        tolerance: Minimum matching ratio required
        
    Returns:
        Validation results
    """
    # Find exact matches
    exact_matches = set(sample_ids_1) & set(sample_ids_2)
    
    # Find fuzzy matches (simplified implementation)
    fuzzy_matches = []
    unmatched_1 = set(sample_ids_1) - exact_matches
    unmatched_2 = set(sample_ids_2) - exact_matches
    
    for id1 in list(unmatched_1):
        for id2 in list(unmatched_2):
            # Simple similarity check
            if id1.lower() == id2.lower() or id1.replace('-', '').replace('_', '') == id2.replace('-', '').replace('_', ''):
                fuzzy_matches.append((id1, id2))
                unmatched_1.discard(id1)
                unmatched_2.discard(id2)
                break
    
    # Calculate statistics
    total_samples = max(len(sample_ids_1), len(sample_ids_2))
    matched_samples = len(exact_matches) + len(fuzzy_matches)
    matching_ratio = matched_samples / total_samples if total_samples > 0 else 0
    
    validation_result = {
        'exact_matches': list(exact_matches),
        'fuzzy_matches': fuzzy_matches,
        'unmatched_samples_1': list(unmatched_1),
        'unmatched_samples_2': list(unmatched_2),
        'matching_ratio': matching_ratio,
        'is_valid': matching_ratio >= tolerance,
        'total_samples': total_samples,
        'matched_samples': matched_samples
    }
    
    logger.info(f"Sample mapping validation: {matched_samples}/{total_samples} matched ({matching_ratio:.2%})")
    
    return validation_result


def create_sample_mapping_report(validation_result: Dict[str, Any], output_path: Optional[Path] = None) -> str:
    """
    Create a detailed sample mapping report.
    
    Args:
        validation_result: Validation results from validate_sample_mapping
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    report = []
    report.append("Sample Mapping Validation Report")
    report.append("=" * 40)
    report.append(f"Total samples: {validation_result['total_samples']}")
    report.append(f"Matched samples: {validation_result['matched_samples']}")
    report.append(f"Matching ratio: {validation_result['matching_ratio']:.2%}")
    report.append(f"Validation status: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
    report.append("")
    
    report.append("Exact Matches:")
    for match in validation_result['exact_matches']:
        report.append(f"  - {match}")
    
    report.append("")
    report.append("Fuzzy Matches:")
    for match1, match2 in validation_result['fuzzy_matches']:
        report.append(f"  - {match1} <-> {match2}")
    
    report.append("")
    report.append("Unmatched Samples (Dataset 1):")
    for sample in validation_result['unmatched_samples_1']:
        report.append(f"  - {sample}")
    
    report.append("")
    report.append("Unmatched Samples (Dataset 2):")
    for sample in validation_result['unmatched_samples_2']:
        report.append(f"  - {sample}")
    
    report_text = "\n".join(report)
    
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Saved sample mapping report to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return report_text


def merge_multi_omics_data(omics_data: Dict[str, pd.DataFrame], 
                          sample_id_column: str = 'sample_id') -> pd.DataFrame:
    """
    Merge multi-omics data by sample ID.
    
    Args:
        omics_data: Dictionary of omics data DataFrames
        sample_id_column: Column name containing sample IDs
        
    Returns:
        Merged DataFrame
    """
    if not omics_data:
        return pd.DataFrame()
    
    # Start with the first dataset
    merged_data = None
    merge_info = []
    
    for data_type, data_df in omics_data.items():
        if data_df.empty:
            logger.warning(f"Empty DataFrame for {data_type}, skipping")
            continue
            
        # Ensure sample_id_column exists
        if sample_id_column not in data_df.columns:
            logger.warning(f"Sample ID column '{sample_id_column}' not found in {data_type}")
            continue
        
        # Set sample_id as index
        data_indexed = data_df.set_index(sample_id_column)
        
        if merged_data is None:
            merged_data = data_indexed.copy()
            merge_info.append(f"{data_type}: {data_indexed.shape}")
        else:
            # Merge with existing data
            original_shape = merged_data.shape
            
            # Find common samples
            common_samples = merged_data.index.intersection(data_indexed.index)
            
            if len(common_samples) == 0:
                logger.warning(f"No common samples between merged data and {data_type}")
                continue
            
            # Merge only common samples
            merged_data = merged_data.loc[common_samples]
            data_to_merge = data_indexed.loc[common_samples]
            
            # Add prefix to column names to avoid conflicts
            data_to_merge = data_to_merge.add_prefix(f"{data_type}_")
            
            # Merge horizontally
            merged_data = pd.concat([merged_data, data_to_merge], axis=1)
            
            new_shape = merged_data.shape
            merge_info.append(f"{data_type}: {original_shape} -> {new_shape} (common: {len(common_samples)})")
    
    if merged_data is None:
        return pd.DataFrame()
    
    logger.info(f"Merged multi-omics data: {merged_data.shape}")
    logger.info(f"Merge details: {'; '.join(merge_info)}")
    
    return merged_data.reset_index()