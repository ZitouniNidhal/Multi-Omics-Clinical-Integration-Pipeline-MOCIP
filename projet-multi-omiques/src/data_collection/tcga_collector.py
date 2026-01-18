"""TCGA data collector using GDC API."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from .base_collector import BaseCollector
from ..exceptions import TCGAError, DataCollectionError

logger = logging.getLogger(__name__)


class TCGLCollector(BaseCollector):
    """Collector for TCGA data using GDC API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TCGA collector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config.get('tcga', {}), config.get('tcga', {}).get('download_dir'))
        
        self.api_endpoint = config.get('tcga', {}).get('gdc_api_endpoint', 'https://api.gdc.cancer.gov')
        self.projects = config.get('tcga', {}).get('projects', [])
        self.data_types = config.get('tcga', {}).get('data_types', ['gene_expression', 'clinical'])
        
        logger.info(f"Initialized TCGA collector with endpoint: {self.api_endpoint}")
    
    def search(self, query: str = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for TCGA datasets.
        
        Args:
            query: Search query (optional)
            **kwargs: Additional search parameters
            
        Returns:
            List of matching datasets
        """
        try:
            # Search for projects
            projects_url = f"{self.api_endpoint}/projects"
            params = {
                'size': 100,
                'fields': 'project_id,name,primary_site,disease_type',
                'format': 'json'
            }
            
            if query:
                params['query'] = query
            
            response = self.safe_api_call(projects_url, params=params)
            data = response.json()
            
            datasets = []
            for project in data.get('data', {}).get('hits', []):
                datasets.append({
                    'id': project['project_id'],
                    'name': project.get('name', ''),
                    'primary_site': project.get('primary_site', ''),
                    'disease_type': project.get('disease_type', ''),
                    'source': 'TCGA'
                })
            
            logger.info(f"Found {len(datasets)} TCGA projects")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to search TCGA datasets: {e}")
            raise TCGAError(f"TCGA search failed: {e}")
    
    def validate_dataset(self, dataset_id: str) -> bool:
        """
        Validate if a TCGA project exists.
        
        Args:
            dataset_id: TCGA project ID (e.g., TCGA-BRCA)
            
        Returns:
            True if project exists
        """
        try:
            project_url = f"{self.api_endpoint}/projects/{dataset_id}"
            response = self.safe_api_call(project_url)
            
            if response.status_code == 200:
                logger.info(f"Validated TCGA project: {dataset_id}")
                return True
            else:
                logger.warning(f"TCGA project not found: {dataset_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate TCGA project {dataset_id}: {e}")
            return False
    
    def collect(self, project: str = None, data_types: List[str] = None, 
                **kwargs) -> Dict[str, Any]:
        """
        Collect TCGA data for specified project and data types.
        
        Args:
            project: TCGA project ID (e.g., TCGA-BRCA)
            data_types: List of data types to collect
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing collected data
        """
        if not project:
            project = kwargs.get('project_id', self.projects[0] if self.projects else None)
        
        if not project:
            raise TCGAError("No TCGA project specified")
        
        if not data_types:
            data_types = self.data_types
        
        logger.info(f"Collecting TCGA data for project: {project}, types: {data_types}")
        
        # Validate project
        if not self.validate_dataset(project):
            raise TCGAError(f"Invalid TCGA project: {project}")
        
        collected_data = {
            'project_id': project,
            'data_types': data_types,
            'collection_timestamp': datetime.now().isoformat(),
            'data': {}
        }
        
        # Collect different data types
        for data_type in data_types:
            try:
                logger.info(f"Collecting {data_type} data for {project}")
                
                if data_type == 'gene_expression':
                    data = self._collect_gene_expression(project, **kwargs)
                elif data_type == 'clinical':
                    data = self._collect_clinical_data(project, **kwargs)
                elif data_type == 'copy_number':
                    data = self._collect_copy_number(project, **kwargs)
                elif data_type == 'mutation':
                    data = self._collect_mutation_data(project, **kwargs)
                else:
                    logger.warning(f"Unsupported data type: {data_type}")
                    continue
                
                collected_data['data'][data_type] = data
                
            except Exception as e:
                logger.error(f"Failed to collect {data_type} data for {project}: {e}")
                # Continue with other data types even if one fails
                continue
        
        # Save metadata
        self.save_metadata(collected_data, f"tcga_{project.lower()}")
        
        logger.info(f"Successfully collected TCGA data for {project}")
        return collected_data
    
    def _collect_gene_expression(self, project: str, **kwargs) -> Dict[str, Any]:
        """Collect gene expression data."""
        try:
            # Search for gene expression files
            files_url = f"{self.api_endpoint}/files"
            params = {
                'filters': json.dumps({
                    'op': 'and',
                    'content': [
                        {'op': '=', 'content': {'field': 'cases.project.project_id', 'value': project}},
                        {'op': '=', 'content': {'field': 'data_type', 'value': 'Gene Expression Quantification'}},
                        {'op': '=', 'content': {'field': 'experimental_strategy', 'value': 'RNA-Seq'}}
                    ]
                }),
                'size': 1000,
                'fields': 'file_id,file_name,cases.case_id,cases.samples.sample_type',
                'format': 'json'
            }
            
            response = self.safe_api_call(files_url, params=params)
            data = response.json()
            
            files = data.get('data', {}).get('hits', [])
            if not files:
                logger.warning(f"No gene expression files found for {project}")
                return {'expression_matrix': pd.DataFrame(), 'metadata': {}}
            
            # Download expression files
            expression_data = []
            metadata = []
            
            for i, file_info in enumerate(files):
                try:
                    file_id = file_info['file_id']
                    file_name = file_info['file_name']
                    case_id = file_info.get('cases', [{}])[0].get('case_id', '')
                    sample_type = file_info.get('cases', [{}])[0].get('samples', [{}])[0].get('sample_type', '')
                    
                    # Download file
                    download_url = f"{self.api_endpoint}/data/{file_id}"
                    local_file = self.download_file(download_url, f"{case_id}_{file_name}")
                    
                    # Process expression data (assuming HTSeq counts format)
                    if local_file.suffix == '.gz':
                        import gzip
                        with gzip.open(local_file, 'rt') as f:
                            expr_data = self._parse_htseq_counts(f)
                    else:
                        with open(local_file, 'r') as f:
                            expr_data = self._parse_htseq_counts(f)
                    
                    # Add to collection
                    expr_data['case_id'] = case_id
                    expr_data['sample_type'] = sample_type
                    expression_data.append(expr_data)
                    
                    metadata.append({
                        'case_id': case_id,
                        'file_id': file_id,
                        'file_name': file_name,
                        'sample_type': sample_type
                    })
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(files)} expression files")
                        
                except Exception as e:
                    logger.error(f"Failed to process expression file {file_info.get('file_name', 'unknown')}: {e}")
                    continue
            
            # Combine expression data
            if expression_data:
                combined_expr = pd.concat(expression_data, ignore_index=True)
                expression_matrix = combined_expr.pivot_table(
                    index='gene_id', columns='case_id', values='count', fill_value=0
                )
            else:
                expression_matrix = pd.DataFrame()
            
            result = {
                'expression_matrix': expression_matrix,
                'metadata': pd.DataFrame(metadata),
                'processing_info': {
                    'num_samples': len(metadata),
                    'num_genes': len(expression_matrix),
                    'sample_types': list(set(m['sample_type'] for m in metadata))
                }
            }
            
            logger.info(f"Collected gene expression data: {expression_matrix.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to collect gene expression data for {project}: {e}")
            raise TCGAError(f"Gene expression collection failed: {e}")
    
    def _collect_clinical_data(self, project: str, **kwargs) -> Dict[str, Any]:
        """Collect clinical data."""
        try:
            # Search for clinical files
            files_url = f"{self.api_endpoint}/files"
            params = {
                'filters': json.dumps({
                    'op': 'and',
                    'content': [
                        {'op': '=', 'content': {'field': 'cases.project.project_id', 'value': project}},
                        {'op': '=', 'content': {'field': 'data_type', 'value': 'Clinical data'}}
                    ]
                }),
                'size': 100,
                'fields': 'file_id,file_name,cases.case_id',
                'format': 'json'
            }
            
            response = self.safe_api_call(files_url, params=params)
            data = response.json()
            
            files = data.get('data', {}).get('hits', [])
            if not files:
                logger.warning(f"No clinical files found for {project}")
                return {'clinical_data': pd.DataFrame(), 'metadata': {}}
            
            # Collect clinical data
            clinical_data = []
            metadata = []
            
            for file_info in files:
                try:
                    file_id = file_info['file_id']
                    file_name = file_info['file_name']
                    case_id = file_info.get('cases', [{}])[0].get('case_id', '')
                    
                    # Download clinical file
                    download_url = f"{self.api_endpoint}/data/{file_id}"
                    local_file = self.download_file(download_url, f"{case_id}_clinical_{file_name}")
                    
                    # Parse clinical data (assuming JSON format)
                    if local_file.suffix == '.json':
                        with open(local_file, 'r') as f:
                            clinical_json = json.load(f)
                        
                        # Extract relevant clinical information
                        patient_info = self._extract_clinical_info(clinical_json, case_id)
                        clinical_data.append(patient_info)
                        
                        metadata.append({
                            'case_id': case_id,
                            'file_id': file_id,
                            'file_name': file_name
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to process clinical file {file_info.get('file_name', 'unknown')}: {e}")
                    continue
            
            # Combine clinical data
            if clinical_data:
                clinical_df = pd.DataFrame(clinical_data)
            else:
                clinical_df = pd.DataFrame()
            
            result = {
                'clinical_data': clinical_df,
                'metadata': pd.DataFrame(metadata),
                'processing_info': {
                    'num_patients': len(clinical_df),
                    'clinical_variables': list(clinical_df.columns) if not clinical_df.empty else []
                }
            }
            
            logger.info(f"Collected clinical data: {clinical_df.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to collect clinical data for {project}: {e}")
            raise TCGAError(f"Clinical data collection failed: {e}")
    
    def _collect_copy_number(self, project: str, **kwargs) -> Dict[str, Any]:
        """Collect copy number variation data."""
        try:
            # Similar implementation to gene expression but for CNV data
            logger.info(f"Collecting copy number data for {project}")
            
            # Placeholder implementation
            return {
                'copy_number_matrix': pd.DataFrame(),
                'metadata': pd.DataFrame(),
                'processing_info': {'num_samples': 0, 'num_regions': 0}
            }
            
        except Exception as e:
            logger.error(f"Failed to collect copy number data for {project}: {e}")
            raise TCGAError(f"Copy number collection failed: {e}")
    
    def _collect_mutation_data(self, project: str, **kwargs) -> Dict[str, Any]:
        """Collect mutation data."""
        try:
            # Similar implementation to gene expression but for mutation data
            logger.info(f"Collecting mutation data for {project}")
            
            # Placeholder implementation
            return {
                'mutation_matrix': pd.DataFrame(),
                'metadata': pd.DataFrame(),
                'processing_info': {'num_samples': 0, 'num_genes': 0}
            }
            
        except Exception as e:
            logger.error(f"Failed to collect mutation data for {project}: {e}")
            raise TCGAError(f"Mutation data collection failed: {e}")
    
    def _parse_htseq_counts(self, file_handle) -> pd.DataFrame:
        """Parse HTSeq count file."""
        data = []
        for line in file_handle:
            if line.startswith('_'):  # Skip technical features
                continue
                
            parts = line.strip().split('\t')
            if len(parts) == 2:
                gene_id, count = parts
                try:
                    count = float(count)
                    data.append({'gene_id': gene_id, 'count': count})
                except ValueError:
                    continue
        
        return pd.DataFrame(data)
    
    def _extract_clinical_info(self, clinical_json: Dict[str, Any], case_id: str) -> Dict[str, Any]:
        """Extract relevant clinical information from JSON."""
        clinical_info = {'case_id': case_id}
        
        # Extract demographic information
        demographic = clinical_json.get('demographic', {})
        clinical_info['age'] = demographic.get('age_at_index', None)
        clinical_info['gender'] = demographic.get('gender', None)
        clinical_info['race'] = demographic.get('race', None)
        clinical_info['ethnicity'] = demographic.get('ethnicity', None)
        
        # Extract diagnosis information
        diagnoses = clinical_json.get('diagnoses', [])
        if diagnoses:
            diagnosis = diagnoses[0]  # Use primary diagnosis
            clinical_info['primary_diagnosis'] = diagnosis.get('primary_diagnosis', None)
            clinical_info['tumor_stage'] = diagnosis.get('ajcc_pathologic_stage', None)
            clinical_info['tumor_grade'] = diagnosis.get('tumor_grade', None)
            clinical_info['vital_status'] = diagnosis.get('vital_status', None)
            clinical_info['days_to_death'] = diagnosis.get('days_to_death', None)
            clinical_info['days_to_last_follow_up'] = diagnosis.get('days_to_last_follow_up', None)
        
        # Extract exposures (risk factors)
        exposures = clinical_json.get('exposures', [])
        if exposures:
            exposure = exposures[0]
            clinical_info['smoking_status'] = exposure.get('tobacco_smoking_status', None)
            clinical_info['alcohol_history'] = exposure.get('alcohol_history', None)
        
        return clinical_info