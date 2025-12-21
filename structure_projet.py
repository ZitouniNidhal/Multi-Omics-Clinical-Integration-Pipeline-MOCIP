#!/usr/bin/env python3
"""
Script d'initialisation de la structure du projet Multi-Omiques
Ce script crée l'arborescence complète du projet avec tous les fichiers nécessaires
"""

import os
import json
from pathlib import Path

def create_project_structure():
    """Crée la structure complète du projet"""
    
    # Structure de base
    base_structure = {
        "src/": {
            "__init__.py": "# Package principal du pipeline multi-omiques",
            "data_collection/": {
                "__init__.py": "",
                "tcga_collector.py": """Module de collecte des données TCGA"""
import pandas as pd
import requests
from typing import Dict, List, Optional

class TCGADataCollector:
    """Collecte les données TCGA via GDC API"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.base_url = "https://api.gdc.cancer.gov"
    
    def download_expression_data(self, file_id: str, output_path: str) -> bool:
        pass
    
    def download_clinical_data(self, case_ids: List[str], output_path: str) -> bool:
        pass
""",
                "geo_collector.py": """Module de collecte des données GEO"""
import pandas as pd
from typing import Dict, List

class GEODataCollector:
    """Collecte les données GEO via NCBI API"""
    
    def __init__(self, geo_id: str):
        self.geo_id = geo_id
    
    def download_series_matrix(self, output_path: str) -> pd.DataFrame:
        """Télécharge la matrice de séries GEO"""
        pass
""",
            },
            "preprocessing/": {
                "__init__.py": "",
                "missing_values.py": """Gestion des valeurs manquantes"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Union, List, Optional

class MissingValueHandler:
    """Gère l'imputation des valeurs manquantes"""
    
    def __init__(self, strategy: str = 'knn', **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def filter_low_quality_features(self, data: pd.DataFrame, 
                                  threshold: float = 0.5) -> pd.DataFrame:
        pass
""",
                "normalization.py": """Normalisation des données omiques"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple

class OmicsNormalizer:
    """Normalise les données omiques selon différentes méthodes"""
    
    def __init__(self, method: str = 'tmm'):
        self.method = method
    
    def tmm_normalization(self, data: pd.DataFrame, 
                         reference_column: Optional[str] = None) -> pd.DataFrame:
        """Normalisation TMM (Trimmed Mean of M-values)"""
        pass
    
    def deseq2_size_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalisation avec DESeq2 size factors"""
        pass
    
    def tpm_transformation(self, data: pd.DataFrame, 
                          gene_lengths: pd.Series) -> pd.DataFrame:
        """Transformation TPM (Transcripts Per Million)"""
        pass
""",
                "quality_control.py": """
"""Contrôle qualité des données"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class QualityControl:
    """Effectue le contrôle qualité des données omiques"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
    
    def check_completeness(self, data: pd.DataFrame) -> Dict[str, float]:
        """Vérifie le pourcentage de données manquantes"""
        pass
    
    def detect_outliers(self, data: pd.DataFrame, 
                       method: str = 'iqr') -> List[str]:
        """Détecte les outliers dans les données"""
        pass
    
    def generate_qc_report(self, data: pd.DataFrame) -> Dict:
        """Génère un rapport de qualité complet"""
        pass
""",
            },
            "integration/": {
                "__init__.py": "",
                "sample_alignment.py": """
"""Alignement des échantillons entre datasets"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz

class SampleAlignment:
    """Aligne les échantillons entre différentes modalités de données"""
    
    def __init__(self, fuzzy_matching: bool = False, threshold: float = 0.9):
        self.fuzzy_matching = fuzzy_matching
        self.threshold = threshold
    
    def align_by_patient_id(self, datasets: Dict[str, pd.DataFrame], 
                          patient_id_columns: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Aligne les datasets par ID patient"""
        pass
    
    def align_by_metadata(self, datasets: Dict[str, pd.DataFrame], 
                         metadata_keys: List[str]) -> Dict[str, pd.DataFrame]:
        """Aligne les datasets par métadonnées"""
        pass
""",
                "data_fusion.py": """
"""Fusion des données multi-modalités"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class MultiOmicsFusion:
    """Fusionne les données de différentes modalités omiques"""
    
    def __init__(self, fusion_method: str = 'horizontal'):
        self.fusion_method = fusion_method
    
    def horizontal_fusion(self, datasets: Dict[str, pd.DataFrame], 
                         sample_key: str = 'patient_id') -> pd.DataFrame:
        """Fusion horizontale des datasets"""
        pass
    
    def vertical_fusion(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Fusion verticale des datasets"""
        pass
    
    def scale_features(self, data: pd.DataFrame, 
                      method: str = 'standard') -> pd.DataFrame:
        """Met à l'échelle les features pour la fusion"""
        pass
""",
            },
            "standardization/": {
                "__init__.py": "",
                "fhir_export.py": """
"""Export vers format FHIR R4"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

class FHIRExporter:
    """Exporte les données vers le format FHIR R4"""
    
    def __init__(self):
        self.fhir_version = "4.0.1"
    
    def create_patient_resource(self, patient_data: Dict) -> Dict:
        """Crée une ressource Patient FHIR"""
        pass
    
    def create_observation_resource(self, observation_data: Dict, 
                                  patient_id: str) -> Dict:
        """Crée une ressource Observation FHIR pour données omiques"""
        pass
    
    def export_to_fhir(self, integrated_data: pd.DataFrame, 
                      output_dir: str) -> bool:
        """Exporte les données intégrées vers FHIR"""
        pass
""",
                "json_export.py": """
"""Export vers format JSON avec schéma"""
import json
import pandas as pd
from typing import Dict, List, Optional

class JSONExporter:
    """Exporte les données vers JSON avec schéma standardisé"""
    
    def __init__(self, schema_version: str = '1.0'):
        self.schema_version = schema_version
    
    def create_schema(self, data: pd.DataFrame) -> Dict:
        """Crée un schéma JSON pour les données"""
        pass
    
    def export_with_schema(self, data: pd.DataFrame, 
                          output_path: str) -> bool:
        """Exporte les données avec leur schéma"""
        pass
""",
            },
            "utils/": {
                "__init__.py": "",
                "config.py": """
"""Gestion de la configuration du pipeline"""
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Gère la configuration du pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis YAML"""
        pass
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Sauvegarde la configuration"""
        pass
""",
                "logger.py": """
"""Gestion des logs du pipeline"""
import logging
from pathlib import Path
from datetime import datetime

class PipelineLogger:
    """Gère la journalisation du pipeline"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logger(log_level)
    
    def setup_logger(self, log_level: str) -> None:
        """Configure le système de logging"""
        pass
    
    def log_step(self, step_name: str, status: str, details: str = "") -> None:
        """Log une étape du pipeline"""
        pass
""",
            },
        },
        "data/": {
            "raw/": {
                ".gitkeep": "# Fichier pour garder le répertoire dans git",
            },
            "processed/": {
                ".gitkeep": "# Fichier pour garder le répertoire dans git",
            },
            "external/": {
                ".gitkeep": "# Fichier pour garder le répertoire dans git",
            },
        },
        "notebooks/": {
            "01_data_exploration.ipynb": """
"""
Notebook d'exploration des données multi-omiques
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour les plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# TODO: Charger les données brutes
# TODO: Analyse exploratoire
# TODO: Visualisations
""",
            "02_data_cleaning_demo.ipynb": """
"""
Démonstration du module de nettoyage de données
"""
import sys
sys.path.append('../src')

from preprocessing.missing_values import MissingValueHandler
from preprocessing.normalization import OmicsNormalizer
from preprocessing.quality_control import QualityControl

# TODO: Démonstration du nettoyage
""",
            "03_data_integration_demo.ipynb": """
"""
Démonstration de l'intégration multi-modalités
"""
import sys
sys.path.append('../src')

from integration.sample_alignment import SampleAlignment
from integration.data_fusion import MultiOmicsFusion

# TODO: Démonstration de l'intégration
""",
        },
        "tests/": {
            "__init__.py": "",
            "test_preprocessing/": {
                "__init__.py": "",
                "test_missing_values.py": """
"""Tests pour le module de gestion des valeurs manquantes"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.missing_values import MissingValueHandler

def test_missing_value_handler_initialization():
    """Test l'initialisation du handler"""
    handler = MissingValueHandler(strategy='knn', k=5)
    assert handler.strategy == 'knn'

def test_knn_imputation():
    """Test l'imputation KNN"""
    # Créer des données de test avec valeurs manquantes
    data = pd.DataFrame({
        'gene1': [1, 2, np.nan, 4, 5],
        'gene2': [2, np.nan, 4, 5, 6]
    })
    
    handler = MissingValueHandler(strategy='knn', k=2)
    result = handler.fit_transform(data)
    
    assert not result.isnull().any().any()
""",
                "test_normalization.py": """
"""Tests pour le module de normalisation"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.normalization import OmicsNormalizer

def test_tmm_normalization():
    """Test la normalisation TMM"""
    data = pd.DataFrame({
        'sample1': [100, 200, 300, 400],
        'sample2': [150, 250, 350, 450],
        'sample3': [120, 220, 320, 420]
    })
    
    normalizer = OmicsNormalizer(method='tmm')
    result = normalizer.tmm_normalization(data)
    
    assert result.shape == data.shape
    assert not result.isnull().any().any()
""",
            },
            "test_integration/": {
                "__init__.py": "",
                "test_sample_alignment.py": """
"""Tests pour l'alignement des échantillons"""
import pytest
import pandas as pd
from src.integration.sample_alignment import SampleAlignment

def test_align_by_patient_id():
    """Test l'alignement par ID patient"""
    omic_data = pd.DataFrame({
        'patient_id': ['P1', 'P2', 'P3'],
        'gene1': [1, 2, 3]
    })
    
    clinical_data = pd.DataFrame({
        'patient_id': ['P1', 'P2', 'P4'],
        'age': [45, 50, 55]
    })
    
    aligner = SampleAlignment()
    result = aligner.align_by_patient_id(
        {'omic': omic_data, 'clinical': clinical_data},
        {'omic': 'patient_id', 'clinical': 'patient_id'}
    )
    
    assert len(result['omic']) == 2  # P1 et P2
    assert len(result['clinical']) == 2  # P1 et P2
""",
            },
        },
        "config/": {
            "config.yaml": """
# Configuration du pipeline multi-omiques

# Paramètres généraux
general:
  project_name: "multi_omics_pipeline"
  version: "1.0.0"
  author: "Votre Nom"
  
# Paramètres de collecte de données
data_collection:
  tcga:
    base_url: "https://api.gdc.cancer.gov"
    data_category: "Transcriptome Profiling"
    data_type: "Gene Expression Quantification"
  geo:
    base_url: "https://www.ncbi.nlm.nih.gov/geo/"
    
# Paramètres de préprocessing
preprocessing:
  missing_values:
    strategy: "knn"
    k: 5
    threshold_missing: 0.5
  
  normalization:
    method: "tmm"
    log_transform: true
    
  quality_control:
    min_samples_per_gene: 10
    max_missing_rate: 0.5
    outlier_method: "iqr"

# Paramètres d'intégration
integration:
  sample_alignment:
    fuzzy_matching: false
    match_threshold: 0.9
  
  data_fusion:
    method: "horizontal"
    scale_features: true

# Paramètres d'export
export:
  fhir:
    version: "R4"
    resource_types: ["Patient", "Observation", "DiagnosticReport"]
  
  json:
    schema_version: "1.0"
    include_metadata: true
  
  csv:
    separator: "\t"
    include_header: true

# Paramètres de logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pipeline.log"
""",
        },
        "docs/": {
            "README.md": """
# Pipeline d'Intégration Multi-Omiques

## Description
Pipeline modulaire pour l'intégration de données multi-omiques et cliniques dans le domaine de la santé.

## Structure du Projet
```
projet-multi-omiques/
├── src/                    # Code source
├── data/                   # Données
├── notebooks/              # Notebooks Jupyter
├── tests/                  # Tests unitaires
├── config/                 # Configuration
└── docs/                   # Documentation
```

## Installation

```bash
# Cloner le repository
git clone https://github.com/votreusername/projet-multi-omiques.git
cd projet-multi-omiques

# Créer l'environnement virtuel
python -m venv omics_env
source omics_env/bin/activate  # Linux/Mac
# ou omics_env\\Scripts\\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

## Utilisation

```python
from src.pipeline import MultiOmicsPipeline

# Initialiser le pipeline
pipeline = MultiOmicsPipeline(config_path="config/config.yaml")

# Exécuter le pipeline
results = pipeline.run(
    omic_data_path="data/raw/expression_data.csv",
    clinical_data_path="data/raw/clinical_data.csv"
)
```

## Tests

```bash
# Exécuter les tests
pytest tests/

# Avec couverture
pytest --cov=src tests/
```

## Documentation

Voir la documentation complète dans le dossier `docs/`.

## Licence

MIT License
""",
            "API.md": """
# Documentation API

## Modules Principaux

### Data Collection
- `TCGADataCollector` : Collecte des données TCGA
- `GEODataCollector` : Collecte des données GEO

### Preprocessing
- `MissingValueHandler` : Gestion des valeurs manquantes
- `OmicsNormalizer` : Normalisation des données
- `QualityControl` : Contrôle qualité

### Integration
- `SampleAlignment` : Alignement des échantillons
- `MultiOmicsFusion` : Fusion multi-modalités

### Standardization
- `FHIRExporter` : Export FHIR R4
- `JSONExporter` : Export JSON avec schéma

## Exemples d'Utilisation

Voir les notebooks dans `notebooks/` pour des exemples détaillés.
""",
        },
        "requirements.txt": """
# Dépendances principales
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0

# Pour les tests
pytest>=6.2.0
pytest-cov>=2.12.0

# Pour le traitement des données
scipy>=1.7.0

# Pour le matching fuzzy (optionnel)
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.0

# Pour les notebooks jupyter
jupyter>=1.0.0
ipykernel>=6.0.0

# Pour la documentation
sphinx>=4.0.0  # Optionnel
""",
        ".gitignore": """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environnements virtuels
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
omics_env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data (trop volumineux pour git)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/external/*
!data/external/.gitkeep

# Logs
logs/
*.log

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Configuration locale
config/local_config.yaml
.env.local

# Résultats des tests
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation générée
docs/_build/
docs/build/
""",
        "setup.py": """
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-omics-pipeline",
    version="1.0.0",
    author="Votre Nom",
    author_email="votre.email@example.com",
    description="Pipeline d'intégration de données multi-omiques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votreusername/projet-multi-omiques",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multi-omics-pipeline=src.cli:main",
        ],
    },
)
""",
    }
    
    # Créer l'arborescence
    for directory, contents in base_structure.items():
        if isinstance(contents, dict):
            # Créer le répertoire
            os.makedirs(directory, exist_ok=True)
            
            # Traiter les sous-répertoires et fichiers
            for item_name, item_content in contents.items():
                item_path = os.path.join(directory, item_name)
                
                if isinstance(item_content, dict):
                    # Sous-répertoire
                    os.makedirs(item_path, exist_ok=True)
                    # Traiter récursivement
                    for sub_item, sub_content in item_content.items():
                        sub_path = os.path.join(item_path, sub_item)
                        if isinstance(sub_content, str):
                            with open(sub_path, 'w', encoding='utf-8') as f:
                                f.write(sub_content.strip())
                else:
                    # Fichier
                    with open(item_path, 'w', encoding='utf-8') as f:
                        f.write(item_content.strip())
        else:
            # Fichier à la racine
            with open(directory, 'w', encoding='utf-8') as f:
                f.write(contents.strip())

def create_main_pipeline():
    """Crée le fichier principal du pipeline"""
    pipeline_content = '''"""
Pipeline principal d'intégration multi-omiques
"""
import pandas as pd
from typing import Dict, Optional, Any
import logging
from pathlib import Path

from .data_collection.tcga_collector import TCGADataCollector
from .data_collection.geo_collector import GEODataCollector
from .preprocessing.missing_values import MissingValueHandler
from .preprocessing.normalization import OmicsNormalizer
from .preprocessing.quality_control import QualityControl
from .integration.sample_alignment import SampleAlignment
from .integration.data_fusion import MultiOmicsFusion
from .standardization.fhir_export import FHIRExporter
from .standardization.json_export import JSONExporter
from .utils.config import ConfigManager
from .utils.logger import PipelineLogger

class MultiOmicsPipeline:
    """Pipeline principal pour l'intégration de données multi-omiques"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigManager(config_path).config
        self.logger = PipelineLogger(
            log_dir=self.config['logging'].get('file', 'logs/pipeline.log')
        )
        
        # Initialiser les modules
        self._init_modules()
    
    def _init_modules(self):
        """Initialise tous les modules du pipeline"""
        self.logger.log_step("INIT", "Initialisation des modules")
        
        # Data collection
        self.tcga_collector = None
        self.geo_collector = None
        
        # Preprocessing
        self.missing_handler = MissingValueHandler(
            strategy=self.config['preprocessing']['missing_values']['strategy'],
            **self.config['preprocessing']['missing_values']
        )
        
        self.normalizer = OmicsNormalizer(
            method=self.config['preprocessing']['normalization']['method']
        )
        
        self.qc = QualityControl(
            thresholds=self.config['preprocessing']['quality_control']
        )
        
        # Integration
        self.aligner = SampleAlignment(
            fuzzy_matching=self.config['integration']['sample_alignment']['fuzzy_matching']
        )
        
        self.fusion = MultiOmicsFusion(
            fusion_method=self.config['integration']['data_fusion']['method']
        )
        
        # Export
        self.fhir_exporter = FHIRExporter()
        self.json_exporter = JSONExporter(
            schema_version=self.config['export']['json']['schema_version']
        )
        
        self.logger.log_step("INIT", "Modules initialisés avec succès")
    
    def run(self, omic_data_path: str, clinical_data_path: str, 
            output_dir: str = "results") -> Dict[str, Any]:
        """Exécute le pipeline complet"""
        
        self.logger.log_step("START", "Démarrage du pipeline")
        
        try:
            # 1. Charger les données
            self.logger.log_step("DATA_LOAD", "Chargement des données")
            omic_data = pd.read_csv(omic_data_path)
            clinical_data = pd.read_csv(clinical_data_path)
            
            # 2. Prétraitement
            self.logger.log_step("PREPROCESS", "Prétraitement des données")
            cleaned_data = self._preprocess_data(omic_data, clinical_data)
            
            # 3. Intégration
            self.logger.log_step("INTEGRATE", "Intégration des données")
            integrated_data = self._integrate_data(cleaned_data)
            
            # 4. Export
            self.logger.log_step("EXPORT", "Export des résultats")
            output_paths = self._export_data(integrated_data, output_dir)
            
            self.logger.log_step("COMPLETE", "Pipeline terminé avec succès")
            
            return {
                'status': 'success',
                'output_paths': output_paths,
                'summary': self._generate_summary(integrated_data)
            }
            
        except Exception as e:
            self.logger.log_step("ERROR", f"Erreur lors de l'exécution: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _preprocess_data(self, omic_data: pd.DataFrame, 
                        clinical_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prétraite les données"""
        
        # Gérer les valeurs manquantes
        omic_clean = self.missing_handler.fit_transform(omic_data)
        clinical_clean = self.missing_handler.fit_transform(clinical_data)
        
        # Normaliser les données omiques
        omic_normalized = self.normalizer.tmm_normalization(omic_clean)
        
        # Contrôle qualité
        omic_qc = self.qc.generate_qc_report(omic_normalized)
        clinical_qc = self.qc.generate_qc_report(clinical_clean)
        
        return {
            'omic': omic_normalized,
            'clinical': clinical_clean,
            'qc_reports': {
                'omic': omic_qc,
                'clinical': clinical_qc
            }
        }
    
    def _integrate_data(self, cleaned_data: Dict) -> pd.DataFrame:
        """Intègre les données multi-modalités"""
        
        omic_data = cleaned_data['omic']
        clinical_data = cleaned_data['clinical']
        
        # Aligner les échantillons
        aligned_data = self.aligner.align_by_patient_id(
            {'omic': omic_data, 'clinical': clinical_data},
            {'omic': 'patient_id', 'clinical': 'patient_id'}
        )
        
        # Fusionner les données
        integrated = self.fusion.horizontal_fusion(aligned_data)
        
        return integrated
    
    def _export_data(self, integrated_data: pd.DataFrame, 
                    output_dir: str) -> Dict[str, str]:
        """Exporte les données dans différents formats"""
        
        output_paths = {}
        
        # Créer le répertoire de sortie
        Path(output_dir).mkdir(exist_ok=True)
        
        # Export FHIR
        fhir_path = f"{output_dir}/fhir_export.json"
        self.fhir_exporter.export_to_fhir(integrated_data, fhir_path)
        output_paths['fhir'] = fhir_path
        
        # Export JSON
        json_path = f"{output_dir}/integrated_data.json"
        self.json_exporter.export_with_schema(integrated_data, json_path)
        output_paths['json'] = json_path
        
        # Export CSV
        csv_path = f"{output_dir}/integrated_data.csv"
        integrated_data.to_csv(csv_path, index=False)
        output_paths['csv'] = csv_path
        
        return output_paths
    
    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un résumé des données traitées"""
        return {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'completeness': 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
        }

# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Multi-Omiques")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--omic-data", required=True)
    parser.add_argument("--clinical-data", required=True)
    parser.add_argument("--output-dir", default="results")
    
    args = parser.parse_args()
    
    pipeline = MultiOmicsPipeline(args.config)
    result = pipeline.run(args.omic_data, args.clinical_data, args.output_dir)
    
    print(json.dumps(result, indent=2))
'''
    
    with open("src/pipeline.py", "w", encoding='utf-8') as f:
        f.write(pipeline_content)

def main():
    """Fonction principale pour créer la structure du projet"""
    print("🧬 Création de la structure du projet Multi-Omiques...")
    print("=" * 60)
    
    try:
        create_project_structure()
        print("✅ Structure de base créée avec succès!")
        
        create_main_pipeline()
        print("✅ Pipeline principal créé!")
        
        print("\\n📁 Structure créée:")
        print("├── src/                    # Code source")
        print("├── data/                   # Données")
        print("├── notebooks/              # Notebooks Jupyter")
        print("├── tests/                  # Tests unitaires")
        print("├── config/                 # Configuration")
        print("└── docs/                   # Documentation")
        
        print("\\n🚀 Prochaines étapes:")
        print("1. Créer l'environnement virtuel: python -m venv omics_env")
        print("2. Activer l'environnement: source omics_env/bin/activate")
        print("3. Installer les dépendances: pip install -r requirements.txt")
        print("4. Configurer le fichier config/config.yaml")
        print("5. Commencer le développement!")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")

if __name__ == "__main__":
    main()