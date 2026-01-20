
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Set
import logging
import requests
import json
from pathlib import Path
import pickle
from ..exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class GeneMapper:
    """Handle gene ID mapping across different databases and species."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize gene mapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.source_id = self.config.get('source_id', 'ensembl')
        self.target_id = self.config.get('target_id', 'entrez')
        self.species = self.config.get('species', 'human')
        self.cache_dir = Path(self.config.get('cache_dir', 'cache/gene_mapping'))
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mapping databases
        self.ensembl_to_entrez = {}
        self.entrez_to_ensembl = {}
        self.symbol_to_ids = {}
        self.id_to_symbol = {}
        
        # Load or build mapping databases
        self._load_mapping_databases()
        
        logger.info(f"Initialized GeneMapper: {self.source_id} -> {self.target_id} ({self.species})")
    
    def map_ids(self, gene_ids: List[str], source_id: str = None, 
                target_id: str = None, species: str = None) -> Dict[str, str]:
        """
        Map gene IDs from source to target format.
        
        Args:
            gene_ids: List of gene IDs to map
            source_id: Source ID type (e.g., 'ensembl', 'entrez', 'symbol')
            target_id: Target ID type
            species: Species name
            
        Returns:
            Dictionary mapping source IDs to target IDs
        """
        if source_id is None:
            source_id = self.source_id
        if target_id is None:
            target_id = self.target_id
        if species is None:
            species = self.species
        
        logger.info(f"Mapping {len(gene_ids)} IDs from {source_id} to {target_id} ({species})")
        
        # Remove duplicates and NAs
        gene_ids = list(set([str(gid) for gid in gene_ids if pd.notna(gid) and gid != '']))
        
        if not gene_ids:
            return {}
        
        try:
            if source_id == target_id:
                # No mapping needed
                return {gid: gid for gid in gene_ids}
            
            # Try different mapping strategies
            if source_id == 'ensembl' and target_id == 'entrez':
                mapping = self._map_ensembl_to_entrez(gene_ids, species)
            elif source_id == 'entrez' and target_id == 'ensembl':
                mapping = self._map_entrez_to_ensembl(gene_ids, species)
            elif source_id == 'symbol' and target_id == 'entrez':
                mapping = self._map_symbol_to_entrez(gene_ids, species)
            elif source_id == 'symbol' and target_id == 'ensembl':
                mapping = self._map_symbol_to_ensembl(gene_ids, species)
            elif source_id == 'entrez' and target_id == 'symbol':
                mapping = self._map_entrez_to_symbol(gene_ids, species)
            elif source_id == 'ensembl' and target_id == 'symbol':
                mapping = self._map_ensembl_to_symbol(gene_ids, species)
            else:
                # Use mygene.info as fallback
                mapping = self._map_using_mygene(gene_ids, source_id, target_id, species)
            
            # Log mapping statistics
            mapped_count = sum(1 for v in mapping.values() if v and v != '')
            logger.info(f"Successfully mapped {mapped_count}/{len(gene_ids)} IDs ({mapped_count/len(gene_ids)*100:.1f}%)")
            
            return mapping
            
        except Exception as e:
            logger.error(f"Gene ID mapping failed: {e}")
            raise PreprocessingError(f"Gene ID mapping failed: {e}")
    
    def map_dataframe(self, df: pd.DataFrame, id_column: str = None, 
                     source_id: str = None, target_id: str = None, 
                     species: str = None) -> pd.DataFrame:
        """
        Map gene IDs in a DataFrame.
        
        Args:
            df: Input DataFrame
            id_column: Column containing gene IDs
            source_id: Source ID type
            target_id: Target ID type
            species: Species name
            
        Returns:
            DataFrame with mapped IDs
        """
        if id_column is None:
            # Try to find ID column
            possible_cols = ['gene_id', 'ensembl_id', 'entrez_id', 'gene_symbol', 'symbol']
            for col in possible_cols:
                if col in df.columns:
                    id_column = col
                    break
            
            if id_column is None:
                raise ValueError("No ID column found in DataFrame")
        
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in DataFrame")
        
        logger.info(f"Mapping gene IDs in DataFrame column: {id_column}")
        
        # Get unique gene IDs
        gene_ids = df[id_column].unique().tolist()
        
        # Map IDs
        mapping = self.map_ids(gene_ids, source_id, target_id, species)
        
        # Create new column with mapped IDs
        new_column_name = f"{target_id or self.target_id}_id"
        df[new_column_name] = df[id_column].map(mapping)
        
        # Remove rows with unmapped IDs
        original_count = len(df)
        df_mapped = df[df[new_column_name].notna() & (df[new_column_name] != '')].copy()
        mapped_count = len(df_mapped)
        
        logger.info(f"Retained {mapped_count}/{original_count} rows after ID mapping")
        
        return df_mapped
    
    def consolidate_gene_annotations(self, gene_ids: List[str], species: str = None) -> pd.DataFrame:
        """
        Consolidate comprehensive gene annotations.
        
        Args:
            gene_ids: List of gene IDs
            species: Species name
            
        Returns:
            DataFrame with consolidated annotations
        """
        if species is None:
            species = self.species
        
        logger.info(f"Consolidating annotations for {len(gene_ids)} genes")
        
        # Create comprehensive mapping
        annotations = []
        
        for gene_id in gene_ids:
            annotation = {
                'input_id': gene_id,
                'ensembl_id': '',
                'entrez_id': '',
                'gene_symbol': '',
                'gene_name': '',
                'chromosome': '',
                'start_position': '',
                'end_position': '',
                'strand': '',
                'gene_type': '',
                'description': ''
            }
            
            try:
                # Get annotations from multiple sources
                ensembl_info = self._get_ensembl_annotation(gene_id, species)
                entrez_info = self._get_entrez_annotation(gene_id, species)
                symbol_info = self._get_symbol_annotation(gene_id, species)
                
                # Combine information
                if ensembl_info:
                    annotation.update(ensembl_info)
                if entrez_info:
                    annotation.update(entrez_info)
                if symbol_info:
                    annotation.update(symbol_info)
                
            except Exception as e:
                logger.warning(f"Failed to get annotation for {gene_id}: {e}")
            
            annotations.append(annotation)
        
        return pd.DataFrame(annotations)
    
    def _load_mapping_databases(self):
        """Load or build gene mapping databases."""
        cache_file = self.cache_dir / f"gene_mapping_{self.species}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    databases = pickle.load(f)
                
                self.ensembl_to_entrez = databases.get('ensembl_to_entrez', {})
                self.entrez_to_ensembl = databases.get('entrez_to_ensembl', {})
                self.symbol_to_ids = databases.get('symbol_to_ids', {})
                self.id_to_symbol = databases.get('id_to_symbol', {})
                
                logger.info(f"Loaded gene mapping databases from cache: {len(self.ensembl_to_entrez)} mappings")
                
            except Exception as e:
                logger.warning(f"Failed to load cached databases: {e}")
                self._build_mapping_databases()
        else:
            self._build_mapping_databases()
    
    def _build_mapping_databases(self):
        """Build gene mapping databases from external sources."""
        logger.info("Building gene mapping databases")
        
        try:
            # Use mygene.info to build comprehensive mapping
            self._build_from_mygene()
            
            # Cache the databases
            self._save_mapping_databases()
            
        except Exception as e:
            logger.error(f"Failed to build mapping databases: {e}")
            # Use minimal built-in mappings as fallback
            self._build_fallback_mappings()
    
    def _build_from_mygene(self):
        """Build mapping databases using mygene.info."""
        try:
            import mygene
            
            mg = mygene.MyGeneInfo()
            
            # Query for all genes in the species
            species_map = {
                'human': 'human',
                'mouse': 'mouse',
                'rat': 'rat'
            }
            
            species_name = species_map.get(self.species, 'human')
            
            # Get gene information
            query_result = mg.query(
                '*',
                species=species_name,
                fields='ensembl.gene,entrezgene,symbol,name,genomic_pos',
                size=50000
            )
            
            # Build mappings
            for hit in query_result.get('hits', []):
                ensembl_id = hit.get('ensembl', {}).get('gene') if isinstance(hit.get('ensembl'), dict) else None
                entrez_id = str(hit.get('entrezgene', ''))
                symbol = hit.get('symbol', '')
                gene_name = hit.get('name', '')
                
                if ensembl_id and entrez_id:
                    self.ensembl_to_entrez[ensembl_id] = entrez_id
                    self.entrez_to_ensembl[entrez_id] = ensembl_id
                
                if symbol:
                    self.symbol_to_ids[symbol.upper()] = {
                        'ensembl': ensembl_id,
                        'entrez': entrez_id,
                        'symbol': symbol,
                        'name': gene_name
                    }
                    
                    if ensembl_id:
                        self.id_to_symbol[ensembl_id] = symbol
                    if entrez_id:
                        self.id_to_symbol[entrez_id] = symbol
            
            logger.info(f"Built mapping databases from mygene.info: {len(self.ensembl_to_entrez)} ensembl-entrez pairs, {len(self.symbol_to_ids)} symbols")
            
        except ImportError:
            logger.warning("mygene not available, using fallback mappings")
            self._build_fallback_mappings()
        except Exception as e:
            logger.warning(f"Failed to build from mygene.info: {e}")
            self._build_fallback_mappings()
    
    def _build_fallback_mappings(self):
        """Build minimal fallback mappings."""
        logger.info("Building fallback mapping databases")
        
        # This would typically load from a local file or use a limited set of known mappings
        # For now, create empty mappings
        self.ensembl_to_entrez = {}
        self.entrez_to_ensembl = {}
        self.symbol_to_ids = {}
        self.id_to_symbol = {}
    
    def _save_mapping_databases(self):
        """Save mapping databases to cache."""
        try:
            databases = {
                'ensembl_to_entrez': self.ensembl_to_entrez,
                'entrez_to_ensembl': self.entrez_to_ensembl,
                'symbol_to_ids': self.symbol_to_ids,
                'id_to_symbol': self.id_to_symbol
            }
            
            cache_file = self.cache_dir / f"gene_mapping_{self.species}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(databases, f)
            
            logger.info(f"Saved mapping databases to cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save mapping databases: {e}")
    
    def _map_ensembl_to_entrez(self, ensembl_ids: List[str], species: str) -> Dict[str, str]:
        """Map Ensembl IDs to Entrez IDs."""
        mapping = {}
        
        for ensembl_id in ensembl_ids:
            entrez_id = self.ensembl_to_entrez.get(ensembl_id, '')
            if entrez_id:
                mapping[ensembl_id] = entrez_id
        
        # Fill in missing mappings using mygene.info
        missing_ids = [gid for gid in ensembl_ids if gid not in mapping]
        if missing_ids:
            additional_mapping = self._map_using_mygene(missing_ids, 'ensembl', 'entrez', species)
            mapping.update(additional_mapping)
        
        return mapping
    
    def _map_entrez_to_ensembl(self, entrez_ids: List[str], species: str) -> Dict[str, str]:
        """Map Entrez IDs to Ensembl IDs."""
        mapping = {}
        
        for entrez_id in entrez_ids:
            ensembl_id = self.entrez_to_ensembl.get(entrez_id, '')
            if ensembl_id:
                mapping[entrez_id] = ensembl_id
        
        # Fill in missing mappings
        missing_ids = [gid for gid in entrez_ids if gid not in mapping]
        if missing_ids:
            additional_mapping = self._map_using_mygene(missing_ids, 'entrez', 'ensembl', species)
            mapping.update(additional_mapping)
        
        return mapping
    
    def _map_symbol_to_entrez(self, symbols: List[str], species: str) -> Dict[str, str]:
        """Map gene symbols to Entrez IDs."""
        mapping = {}
        
        for symbol in symbols:
            symbol_upper = symbol.upper()
            if symbol_upper in self.symbol_to_ids:
                entrez_id = self.symbol_to_ids[symbol_upper].get('entrez', '')
                if entrez_id:
                    mapping[symbol] = entrez_id
        
        return mapping
    
    def _map_symbol_to_ensembl(self, symbols: List[str], species: str) -> Dict[str, str]:
        """Map gene symbols to Ensembl IDs."""
        mapping = {}
        
        for symbol in symbols:
            symbol_upper = symbol.upper()
            if symbol_upper in self.symbol_to_ids:
                ensembl_id = self.symbol_to_ids[symbol_upper].get('ensembl', '')
                if ensembl_id:
                    mapping[symbol] = ensembl_id
        
        return mapping
    
    def _map_entrez_to_symbol(self, entrez_ids: List[str], species: str) -> Dict[str, str]:
        """Map Entrez IDs to gene symbols."""
        mapping = {}
        
        for entrez_id in entrez_ids:
            symbol = self.id_to_symbol.get(entrez_id, '')
            if symbol:
                mapping[entrez_id] = symbol
        
        return mapping
    
    def _map_ensembl_to_symbol(self, ensembl_ids: List[str], species: str) -> Dict[str, str]:
        """Map Ensembl IDs to gene symbols."""
        mapping = {}
        
        for ensembl_id in ensembl_ids:
            symbol = self.id_to_symbol.get(ensembl_id, '')
            if symbol:
                mapping[ensembl_id] = symbol
        
        return mapping
    
    def _map_using_mygene(self, gene_ids: List[str], source_id: str, 
                         target_id: str, species: str) -> Dict[str, str]:
        """Map IDs using mygene.info API."""
        mapping = {}
        
        try:
            import mygene
            
            mg = mygene.MyGeneInfo()
            
            # Determine query field
            query_field = {
                'ensembl': 'ensembl.gene',
                'entrez': 'entrezgene',
                'symbol': 'symbol'
            }.get(source_id, source_id)
            
            # Determine target field
            target_field = {
                'ensembl': 'ensembl.gene',
                'entrez': 'entrezgene',
                'symbol': 'symbol'
            }.get(target_id, target_id)
            
            # Query mygene.info
            result = mg.querymany(
                gene_ids,
                scopes=query_field,
                fields=target_field,
                species=species
            )
            
            # Build mapping
            for item in result:
                if 'query' in item and target_field in item:
                    source_id_value = item['query']
                    target_id_value = item[target_field]
                    
                    # Handle list responses
                    if isinstance(target_id_value, list):
                        target_id_value = target_id_value[0] if target_id_value else ''
                    elif isinstance(target_id_value, dict):
                        target_id_value = target_id_value.get('gene', '')
                    
                    if source_id_value and target_id_value:
                        mapping[source_id_value] = str(target_id_value)
            
            logger.info(f"Mapped {len(mapping)} IDs using mygene.info")
            
        except ImportError:
            logger.warning("mygene not available for mapping")
        except Exception as e:
            logger.warning(f"mygene.info query failed: {e}")
        
        return mapping
    
    def _get_ensembl_annotation(self, gene_id: str, species: str) -> Optional[Dict[str, Any]]:
        """Get Ensembl annotation for a gene."""
        try:
            # Use Ensembl REST API
            server = "https://rest.ensembl.org"
            ext = f"/lookup/id/{gene_id}"
            
            response = requests.get(
                server + ext,
                headers={"Content-Type": "application/json"}
            )
            
            if response.ok:
                data = response.json()
                return {
                    'ensembl_id': gene_id,
                    'gene_name': data.get('display_name', ''),
                    'gene_symbol': data.get('display_name', ''),
                    'chromosome': data.get('seq_region_name', ''),
                    'start_position': data.get('start', ''),
                    'end_position': data.get('end', ''),
                    'strand': data.get('strand', ''),
                    'gene_type': data.get('biotype', ''),
                    'description': data.get('description', '')
                }
            
        except Exception as e:
            logger.debug(f"Failed to get Ensembl annotation for {gene_id}: {e}")
        
        return None
    
    def _get_entrez_annotation(self, gene_id: str, species: str) -> Optional[Dict[str, Any]]:
        """Get Entrez annotation for a gene."""
        try:
            # Use NCBI E-utilities
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            
            # Search for gene
            search_url = f"{base_url}/esearch.fcgi"
            search_params = {
                'db': 'gene',
                'term': gene_id,
                'retmode': 'json'
            }
            
            search_response = requests.get(search_url, params=search_params)
            if not search_response.ok:
                return None
            
            search_data = search_response.json()
            gene_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not gene_ids:
                return None
            
            # Get gene summary
            summary_url = f"{base_url}/esummary.fcgi"
            summary_params = {
                'db': 'gene',
                'id': gene_ids[0],
                'retmode': 'json'
            }
            
            summary_response = requests.get(summary_url, params=summary_params)
            if not summary_response.ok:
                return None
            
            summary_data = summary_response.json()
            gene_data = summary_data.get('result', {}).get(gene_ids[0], {})
            
            return {
                'entrez_id': gene_id,
                'gene_name': gene_data.get('name', ''),
                'gene_symbol': gene_data.get('name', ''),
                'description': gene_data.get('description', '')
            }
            
        except Exception as e:
            logger.debug(f"Failed to get Entrez annotation for {gene_id}: {e}")
        
        return None
    
    def _get_symbol_annotation(self, gene_id: str, species: str) -> Optional[Dict[str, Any]]:
        """Get annotation for a gene symbol."""
        # Try to get from existing mapping
        symbol_upper = gene_id.upper()
        if symbol_upper in self.symbol_to_ids:
            info = self.symbol_to_ids[symbol_upper]
            return {
                'gene_symbol': info['symbol'],
                'gene_name': info.get('name', ''),
                'ensembl_id': info['ensembl'],
                'entrez_id': info['entrez']
            }
        
        return None
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about the mapping databases."""
        return {
            'species': self.species,
            'source_id_type': self.source_id,
            'target_id_type': self.target_id,
            'ensembl_to_entrez_mappings': len(self.ensembl_to_entrez),
            'entrez_to_ensembl_mappings': len(self.entrez_to_ensembl),
            'symbol_to_ids_mappings': len(self.symbol_to_ids),
            'id_to_symbol_mappings': len(self.id_to_symbol),
            'cache_directory': str(self.cache_dir)
        }