# projet-multi-omiques/src/config.py

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from .exceptions import ConfigurationError


class DataCollectionConfig(BaseModel):
    """Configuration for data collection modules."""
    
    tcga: Dict[str, Any] = Field(default_factory=dict)
    geo: Dict[str, Any] = Field(default_factory=dict)
    arrayexpress: Dict[str, Any] = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing modules."""
    
    missing_data: Dict[str, Any] = Field(default_factory=dict)
    normalization: Dict[str, Any] = Field(default_factory=dict)
    quality_control: Dict[str, Any] = Field(default_factory=dict)
    gene_mapping: Dict[str, Any] = Field(default_factory=dict)


class IntegrationConfig(BaseModel):
    """Configuration for integration modules."""
    
    methods: list = Field(default_factory=list)
    dimensionality_reduction: Dict[str, Any] = Field(default_factory=dict)
    sample_alignment: Dict[str, Any] = Field(default_factory=dict)


class ExportConfig(BaseModel):
    """Configuration for export modules."""
    
    formats: list = Field(default_factory=list)
    fhir: Dict[str, Any] = Field(default_factory=dict)
    ml_ready: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Main configuration class for the pipeline."""
    
    pipeline: Dict[str, Any] = Field(default_factory=dict)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('pipeline')
    def validate_pipeline(cls, v):
        """Validate pipeline configuration."""
        if 'name' not in v:
            raise ConfigurationError("Pipeline name is required")
        if 'output_dir' not in v:
            v['output_dir'] = 'data'
        return v
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            return cls(**config_dict)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
                
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (dot notation supported)."""
        keys = key.split('.')
        value = self.dict()
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        config_dict = self.dict()
        
        for key, value in updates.items():
            keys = key.split('.')
            current = config_dict
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        # Re-validate configuration
        new_config = self.__class__(**config_dict)
        for field_name, field_value in new_config:
            setattr(self, field_name, field_value)