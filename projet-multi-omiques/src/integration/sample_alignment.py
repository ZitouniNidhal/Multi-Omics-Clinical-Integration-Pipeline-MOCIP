"""Sample alignment and matching across different omics modalities."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from difflib import SequenceMatcher
import re
import logging
from ..exceptions import IntegrationError, SampleAlignmentError

logger = logging.getLogger(__name__)


class SampleAligner:
    """Align samples across different omics datasets."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize sample aligner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.method = self.config.get('method', 'match_by_patient_id')
        self.fuzzy_matching = self.config.get('fuzzy_matching', True)
        self.tolerance = self.config.get('tolerance', 0.8)
        self.id_patterns = self.config.get('id_patterns', [])
        
        logger.info(f"Initialized SampleAligner with method: {self.method}")
    
    def align_samples(self, omics_data: Dict[str, pd.DataFrame], 
                     sample_id_columns: Optional[Dict[str, str]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Align samples across omics datasets.
        
        Args:
            omics_data: Dictionary of omics DataFrames
            sample_id_columns: Mapping of omics types to sample ID columns
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing aligned data and mapping information
        """
        logger.info("Aligning samples across omics datasets")
        
        # Determine sample ID columns for each omics type
        if sample_id_columns is None:
            sample_id_columns = self._identify_sample_id_columns(omics_data)
        
        # Extract sample IDs from each dataset
        sample_ids = {}
        for omics_type, df in omics_data.items():
            id_column = sample_id_columns.get(omics_type, 'sample_id')
            
            if id_column not in df.columns:
                # Try to use index if column not found
                if df.index.name == id_column or 'sample' in str(df.index.name).lower():
                    sample_ids[omics_type] = df.index.tolist()
                else:
                    logger.warning(f"Sample ID column '{id_column}' not found in {omics_type}")
                    sample_ids[omics_type] = []
            else:
                sample_ids[omics_type] = df[id_column].tolist()
            
            logger.info(f"{omics_type}: {len(sample_ids[omics_type])} samples")
        
        # Find matching samples based on method
        if self.method == 'match_by_patient_id':
            matched_samples = self._match_by_patient_id(sample_ids, **kwargs)
        elif self.method == 'match_by_sample_barcode':
            matched_samples = self._match_by_sample_barcode(sample_ids, **kwargs)
        elif self.method == 'fuzzy_matching':
            matched_samples = self._fuzzy_sample_matching(sample_ids, **kwargs)
        elif self.method == 'time_based_matching':
            matched_samples = self._time_based_matching(sample_ids, **kwargs)
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")
        
        if not matched_samples:
            raise SampleAlignmentError("No matching samples found")
        
        # Create aligned datasets
        aligned_data = self._create_aligned_datasets(omics_data, matched_samples, sample_id_columns)
        
        # Generate alignment report
        alignment_report = self._generate_alignment_report(sample_ids, matched_samples, aligned_data)
        
        result = {
            'aligned_data': aligned_data,
            'matched_samples': matched_samples,
            'sample_id_columns': sample_id_columns,
            'alignment_statistics': self._calculate_alignment_statistics(sample_ids, matched_samples),
            'alignment_report': alignment_report,
            'method_used': self.method
        }
        
        logger.info(f"Sample alignment complete: {len(matched_samples)} matched samples")
        
        return result
    
    def _identify_sample_id_columns(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Automatically identify sample ID columns in each dataset."""
        sample_id_columns = {}
        
        possible_id_patterns = [
            'sample_id', 'sample', 'patient_id', 'patient', 'subject_id', 'subject',
            'sample_name', 'sample barcode', 'aliquot_id', 'aliquot',
            'tcga_id', 'tcga_sample', 'gdc_id', 'case_id'
        ]
        
        for omics_type, df in omics_data.items():
            found_column = None
            
            # Check for exact matches first
            for pattern in possible_id_patterns:
                if pattern in df.columns:
                    found_column = pattern
                    break
            
            # Check for case-insensitive matches
            if found_column is None:
                df_columns_lower = [col.lower() for col in df.columns]
                for pattern in possible_id_patterns:
                    if pattern.lower() in df_columns_lower:
                        idx = df_columns_lower.index(pattern.lower())
                        found_column = df.columns[idx]
                        break
            
            # Check if index might contain sample IDs
            if found_column is None and df.index.name is not None:
                index_name_lower = df.index.name.lower()
                if any(pattern in index_name_lower for pattern in ['sample', 'patient', 'subject']):
                    found_column = df.index.name
            
            # Default to first column if nothing found
            if found_column is None and len(df.columns) > 0:
                found_column = df.columns[0]
                logger.warning(f"Using first column '{found_column}' as sample ID for {omics_type}")
            
            if found_column:
                sample_id_columns[omics_type] = found_column
                logger.info(f"Identified sample ID column for {omics_type}: {found_column}")
        
        return sample_id_columns
    
    def _match_by_patient_id(self, sample_ids: Dict[str, List[str]], **kwargs) -> List[str]:
        """Match samples by extracting patient IDs."""
        logger.info("Matching samples by patient ID")
        
        # Extract patient IDs from sample IDs
        patient_ids = {}
        
        for omics_type, ids in sample_ids.items():
            patient_ids[omics_type] = []
            
            for sample_id in ids:
                patient_id = self._extract_patient_id(sample_id)
                if patient_id:
                    patient_ids[omics_type].append(patient_id)
                else:
                    # Keep original ID if patient ID cannot be extracted
                    patient_ids[omics_type].append(sample_id)
        
        # Find common patient IDs
        common_patient_ids = None
        
        for omics_type, pids in patient_ids.items():
            current_ids = set(pids)
            
            if common_patient_ids is None:
                common_patient_ids = current_ids
            else:
                common_patient_ids = common_patient_ids.intersection(current_ids)
        
        if not common_patient_ids:
            logger.warning("No common patient IDs found")
            return []
        
        # Create sample mapping
        matched_samples = list(common_patient_ids)
        
        logger.info(f"Found {len(matched_samples)} common patient IDs")
        
        return matched_samples
    
    def _extract_patient_id(self, sample_id: str) -> Optional[str]:
        """Extract patient ID from sample ID using common patterns."""
        if not sample_id or pd.isna(sample_id):
            return None
        
        sample_id = str(sample_id).strip()
        
        # Common TCGA patterns
        tcga_pattern = r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})'
        match = re.search(tcga_pattern, sample_id, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # General patient ID patterns
        # Pattern: Patient ID followed by sample codes
        general_patterns = [
            r'([A-Z]{2,3}[0-9]{2,4})',  # Institution code + numbers
            r'([A-Z]{2,3}-[0-9]{2,4})',  # Institution code + numbers with dash
            r'(P[0-9]{3,6})',            # P + numbers
            r'(Patient[0-9]{2,4})',     # Patient + numbers
            r'([0-9]{4,8})'              # Just numbers (4-8 digits)
        ]
        
        for pattern in general_patterns:
            match = re.search(pattern, sample_id, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, return the original ID
        return sample_id
    
    def _match_by_sample_barcode(self, sample_ids: Dict[str, List[str]], **kwargs) -> List[str]:
        """Match samples by TCGA-style barcodes."""
        logger.info("Matching samples by sample barcode")
        
        # Extract TCGA barcodes
        barcodes = {}
        
        for omics_type, ids in sample_ids.items():
            barcodes[omics_type] = []
            
            for sample_id in ids:
                barcode = self._extract_tcga_barcode(sample_id)
                if barcode:
                    barcodes[omics_type].append(barcode)
                else:
                    # Try fuzzy matching if enabled
                    if self.fuzzy_matching:
                        barcodes[omics_type].append(sample_id)
                    else:
                        barcodes[omics_type].append(None)
        
        # Find common barcodes
        common_barcodes = None
        
        for omics_type, bcodes in barcodes.items():
            valid_bcodes = [b for b in bcodes if b is not None]
            current_bcodes = set(valid_bcodes)
            
            if common_barcodes is None:
                common_barcodes = current_bcodes
            else:
                common_barcodes = common_barcodes.intersection(current_bcodes)
        
        if not common_barcodes:
            logger.warning("No common barcodes found")
            return []
        
        matched_samples = list(common_barcodes)
        
        logger.info(f"Found {len(matched_samples)} common barcodes")
        
        return matched_samples
    
    def _extract_tcga_barcode(self, sample_id: str) -> Optional[str]:
        """Extract TCGA barcode from sample ID."""
        if not sample_id or pd.isna(sample_id):
            return None
        
        sample_id = str(sample_id).strip()
        
        # TCGA barcode pattern
        # Format: TCGA-XX-XXXX-XX-X-XX-XXXX
        tcga_pattern = r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9]{2})?(?:-[A-Z0-9])?(?:-[A-Z0-9]{2})?(?:-[A-Z0-9]{4})?)'
        
        match = re.search(tcga_pattern, sample_id, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None
    
    def _fuzzy_sample_matching(self, sample_ids: Dict[str, List[str]], **kwargs) -> List[str]:
        """Perform fuzzy matching of sample IDs."""
        logger.info("Performing fuzzy sample matching")
        
        # Create all possible pairs
        all_ids = set()
        for ids in sample_ids.values():
            all_ids.update(ids)
        
        # Calculate similarity matrix
        similarity_matrix = {}
        
        for id1 in all_ids:
            similarity_matrix[id1] = {}
            for id2 in all_ids:
                if id1 != id2:
                    similarity = SequenceMatcher(None, str(id1), str(id2)).ratio()
                    similarity_matrix[id1][id2] = similarity
        
        # Find fuzzy matches
        matched_groups = []
        used_ids = set()
        
        for omics_type, ids in sample_ids.items():
            for sample_id in ids:
                if sample_id in used_ids:
                    continue
                
                # Find similar IDs
                similar_ids = []
                for other_id, similarity in similarity_matrix.get(sample_id, {}).items():
                    if similarity >= self.tolerance:
                        similar_ids.append((other_id, similarity))
                
                # Create a matched group
                if similar_ids:
                    group = [sample_id] + [id_sim[0] for id_sim in similar_ids]
                    matched_groups.append(group)
                    used_ids.update(group)
        
        # Convert groups to representative IDs (first in group)
        matched_samples = [group[0] for group in matched_groups]
        
        logger.info(f"Found {len(matched_samples)} fuzzy-matched sample groups")
        
        return matched_samples
    
    def _time_based_matching(self, sample_ids: Dict[str, List[str]], 
                           time_info: Optional[Dict[str, pd.Series]] = None, 
                           **kwargs) -> List[str]:
        """Match samples based on time proximity."""
        logger.info("Performing time-based matching")
        
        if time_info is None:
            logger.warning("No time information provided for time-based matching")
            return []
        
        # This is a simplified implementation
        # In practice, you would need proper time parsing and matching logic
        
        matched_samples = []
        
        # For now, just return samples that have time information
        for omics_type, ids in sample_ids.items():
            if omics_type in time_info:
                valid_times = time_info[omics_type].dropna()
                valid_samples = set(valid_times.index)
                
                matched_in_omics = [sample_id for sample_id in ids if sample_id in valid_samples]
                if not matched_samples:
                    matched_samples = set(matched_in_omics)
                else:
                    matched_samples = matched_samples.intersection(set(matched_in_omics))
        
        return list(matched_samples) if matched_samples else []
    
    def _create_aligned_datasets(self, omics_data: Dict[str, pd.DataFrame], 
                                matched_samples: List[str], 
                                sample_id_columns: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Create aligned datasets with only matched samples."""
        logger.info("Creating aligned datasets")
        
        aligned_data = {}
        
        for omics_type, df in omics_data.items():
            id_column = sample_id_columns.get(omics_type, 'sample_id')
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {omics_type}")
                continue
            
            # Filter to matched samples
            if id_column in df.columns:
                filtered_df = df[df[id_column].isin(matched_samples)].copy()
                
                # Sort by sample ID for consistency
                filtered_df = filtered_df.sort_values(id_column)
            else:
                # Try to filter by index
                filtered_df = df[df.index.isin(matched_samples)].copy()
                filtered_df = filtered_df.sort_index()
            
            aligned_data[omics_type] = filtered_df
            
            logger.info(f"Aligned {omics_type}: {len(filtered_df)} samples")
        
        return aligned_data
    
    def _calculate_alignment_statistics(self, sample_ids: Dict[str, List[str]], 
                                      matched_samples: List[str]) -> Dict[str, Any]:
        """Calculate alignment statistics."""
        stats = {
            'total_samples_per_omics': {},
            'matched_samples': len(matched_samples),
            'alignment_percentage': {},
            'missing_samples': {},
            'overlap_matrix': {}
        }
        
        # Calculate statistics for each omics type
        for omics_type, ids in sample_ids.items():
            total_samples = len(ids)
            matched_in_omics = sum(1 for sample_id in ids if sample_id in matched_samples)
            alignment_pct = (matched_in_omics / total_samples * 100) if total_samples > 0 else 0
            
            stats['total_samples_per_omics'][omics_type] = total_samples
            stats['alignment_percentage'][omics_type] = alignment_pct
            stats['missing_samples'][omics_type] = total_samples - matched_in_omics
        
        # Calculate pairwise overlaps
        omics_types = list(sample_ids.keys())
        for i, omics1 in enumerate(omics_types):
            for j, omics2 in enumerate(omics_types):
                if i < j:
                    overlap = len(set(sample_ids[omics1]) & set(sample_ids[omics2]))
                    stats['overlap_matrix'][f"{omics1}_vs_{omics2}"] = overlap
        
        return stats
    
    def _generate_alignment_report(self, sample_ids: Dict[str, List[str]], 
                                 matched_samples: List[str], 
                                 aligned_data: Dict[str, pd.DataFrame]) -> str:
        """Generate detailed alignment report."""
        report = []
        
        report.append("SAMPLE ALIGNMENT REPORT")
        report.append("=" * 50)
        report.append(f"Alignment Method: {self.method}")
        report.append(f"Fuzzy Matching: {self.fuzzy_matching}")
        report.append(f"Tolerance: {self.tolerance}")
        report.append("")
        
        # Overall statistics
        stats = self._calculate_alignment_statistics(sample_ids, matched_samples)
        
        report.append(f"Total Matched Samples: {stats['matched_samples']}")
        report.append("")
        
        # Per-omics statistics
        report.append("Per-Omics Statistics:")
        for omics_type in sample_ids.keys():
            total = stats['total_samples_per_omics'][omics_type]
            matched_pct = stats['alignment_percentage'][omics_type]
            missing = stats['missing_samples'][omics_type]
            
            report.append(f"  {omics_type}:")
            report.append(f"    Total samples: {total}")
            report.append(f"    Matched samples: {total - missing} ({matched_pct:.1f}%)")
            report.append(f"    Missing samples: {missing}")
        
        report.append("")
        
        # Overlap matrix
        if stats['overlap_matrix']:
            report.append("Pairwise Sample Overlaps:")
            for pair, overlap in stats['overlap_matrix'].items():
                report.append(f"  {pair}: {overlap} samples")
        
        report.append("")
        
        # Aligned data summary
        report.append("Aligned Data Summary:")
        for omics_type, df in aligned_data.items():
            report.append(f"  {omics_type}: {df.shape[0]} samples × {df.shape[1]} features")
        
        return "\n".join(report)
    
    def find_sample_mappings(self, sample_ids: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
        """
        Find detailed mappings between sample IDs across omics types.
        
        Args:
            sample_ids: Dictionary of sample IDs per omics type
            
        Returns:
            Dictionary containing sample mappings
        """
        mappings = {}
        
        omics_types = list(sample_ids.keys())
        
        for i, source_omics in enumerate(omics_types):
            mappings[source_omics] = {}
            
            for j, target_omics in enumerate(omics_types):
                if i == j:
                    # Same omics type - identity mapping
                    mappings[source_omics][target_omics] = {
                        sample_id: sample_id for sample_id in sample_ids[source_omics]
                    }
                else:
                    # Cross-omics mapping
                    source_ids = set(sample_ids[source_omics])
                    target_ids = set(sample_ids[target_omics])
                    
                    # Find exact matches
                    exact_matches = source_ids & target_ids
                    
                    # Find fuzzy matches for remaining IDs
                    mapping = {}
                    
                    for source_id in source_ids:
                        if source_id in exact_matches:
                            mapping[source_id] = source_id
                        elif self.fuzzy_matching:
                            # Find best fuzzy match
                            best_match = None
                            best_score = 0
                            
                            for target_id in target_ids:
                                score = SequenceMatcher(None, str(source_id), str(target_id)).ratio()
                                if score > best_score and score >= self.tolerance:
                                    best_score = score
                                    best_match = target_id
                            
                            if best_match:
                                mapping[source_id] = best_match
                    
                    mappings[source_omics][target_omics] = mapping
        
        return mappings
    
    def validate_sample_consistency(self, aligned_data: Dict[str, pd.DataFrame], 
                                  sample_id_columns: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate consistency of sample IDs across aligned datasets.
        
        Args:
            aligned_data: Dictionary of aligned DataFrames
            sample_id_columns: Mapping of sample ID columns
            
        Returns:
            Validation results
        """
        validation_results = {
            'consistent': True,
            'errors': [],
            'warnings': [],
            'sample_order_consistency': {},
            'sample_content_consistency': {}
        }
        
        # Get sample IDs from each dataset
        sample_ids = {}
        
        for omics_type, df in aligned_data.items():
            id_column = sample_id_columns.get(omics_type, 'sample_id')
            
            if id_column in df.columns:
                sample_ids[omics_type] = df[id_column].tolist()
            else:
                sample_ids[omics_type] = df.index.tolist()
        
        # Check sample order consistency
        omics_types = list(sample_ids.keys())
        
        if len(omics_types) > 1:
            reference_omics = omics_types[0]
            reference_ids = sample_ids[reference_omics]
            
            for omics_type in omics_types[1:]:
                current_ids = sample_ids[omics_type]
                
                if len(reference_ids) != len(current_ids):
                    validation_results['consistent'] = False
                    validation_results['errors'].append(
                        f"Different number of samples: {reference_omics} has {len(reference_ids)}, "
                        f"{omics_type} has {len(current_ids)}"
                    )
                    continue
                
                if reference_ids != current_ids:
                    # Check if they contain the same samples (different order)
                    if set(reference_ids) == set(current_ids):
                        validation_results['warnings'].append(
                            f"Sample order differs between {reference_omics} and {omics_type}"
                        )
                        validation_results['sample_order_consistency'][f"{reference_omics}_vs_{omics_type}"] = False
                    else:
                        validation_results['consistent'] = False
                        validation_results['errors'].append(
                            f"Different samples between {reference_omics} and {omics_type}"
                        )
                        validation_results['sample_content_consistency'][f"{reference_omics}_vs_{omics_type}"] = False
                else:
                    validation_results['sample_order_consistency'][f"{reference_omics}_vs_{omics_type}"] = True
                    validation_results['sample_content_consistency'][f"{reference_omics}_vs_{omics_type}"] = True
        
        return validation_results