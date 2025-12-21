#!/usr/bin/env python3
"""
Script simplifié pour créer la structure du projet Multi-Omiques
"""

import os

def create_simple_structure():
    """Crée la structure de base du projet"""
    
    # Structure de répertoires
    directories = [
        "src/data_collection",
        "src/preprocessing", 
        "src/integration",
        "src/standardization",
        "src/utils",
        "data/raw",
        "data/processed", 
        "data/external",
        "notebooks",
        "tests",
        "config",
        "docs",
        "logs"
    ]
    
    # Créer les répertoires
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Créé : {directory}")
    
    # Créer les fichiers __init__.py
    init_files = [
        "src/__init__.py",
        "src/data_collection/__init__.py",
        "src/preprocessing/__init__.py",
        "src/integration/__init__.py", 
        "src/standardization/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# Fichier d'initialisation du package\n")
        print(f"✅ Créé : {init_file}")
    
    # Créer des fichiers .gitkeep pour les répertoires de données
    gitkeep_files = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "data/external/.gitkeep",
        "logs/.gitkeep"
    ]
    
    for gitkeep_file in gitkeep_files:
        with open(gitkeep_file, 'w') as f:
            f.write("# Fichier pour garder le répertoire dans git\n")
        print(f"✅ Créé : {gitkeep_file}")
    
    # Créer les fichiers principaux
    files_to_create = {
        "requirements.txt": """# Dépendances principales
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0

# Pour les tests
pytest>=6.2.0
pytest-cov>=2.12.0

# Pour les notebooks
jupyter>=1.0.0
ipykernel>=6.0.0

# Pour le traitement des données
scipy>=1.7.0
""",
        
        "config/config.yaml": """# Configuration du pipeline multi-omiques
general:
  project_name: "multi_omics_pipeline"
  version: "1.0.0"
  author: "Votre Nom"
  
data_collection:
  tcga:
    base_url: "https://api.gdc.cancer.gov"
  geo:
    base_url: "https://www.ncbi.nlm.nih.gov/geo/"

preprocessing:
  missing_values:
    strategy: "knn"
    k: 5
  normalization:
    method: "tmm"
  quality_control:
    threshold: 0.5

integration:
  sample_alignment:
    method: "patient_id"
  data_fusion:
    method: "horizontal"

export:
  formats: ["fhir", "json", "csv"]
  output_dir: "results"

logging:
  level: "INFO"
  file: "logs/pipeline.log"
""",
        
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
develop-eggs/
dist/
*.egg-info/

# Environnements virtuels
.env
.venv/
omics_env/

# IDE
.vscode/
.idea/

# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Logs
logs/*.log

# Jupyter
.ipynb_checkpoints/
""",
        
        "README.md": """# Pipeline Multi-Omiques

## Description
Pipeline modulaire pour l'intégration de données multi-omiques et cliniques.

## Installation
```bash
python -m venv omics_env
source omics_env/bin/activate
pip install -r requirements.txt
```

## Utilisation
```python
# TODO: Exemple d'utilisation
```

## Structure
- `src/` : Code source
- `data/` : Données
- `notebooks/` : Notebooks Jupyter
- `tests/` : Tests unitaires
- `config/` : Configuration

## Auteur
Votre Nom - Projet Multi-Omiques
```

        "src/pipeline.py": "#!/usr/bin/env python3\n\"\"\"\nPipeline principal multi-omiques\n\"\"\"\nimport pandas as pd\nimport yaml\nfrom pathlib import Path\n\nclass MultiOmicsPipeline:\n    def __init__(self, config_path=\"config/config.yaml\"):\n        with open(config_path, 'r') as f:\n            self.config = yaml.safe_load(f)\n        \n        print(f\"Pipeline initialisé : {self.config['general']['project_name']}\")\n    \n    def run(self, omic_data_path, clinical_data_path):\n        print(f\"Exécution du pipeline sur {omic_data_path} et {clinical_data_path}\")\n        # TODO: Implémenter le pipeline complet\n        pass\n\nif __name__ == \"__main__\":\n    pipeline = MultiOmicsPipeline()\n    pipeline.run(\"data/raw/expression.csv\", \"data/raw/clinical.csv\")\n",

        "notebooks/01_data_exploration.ipynb": "{\n \"cells\": [\n  {\n   \"cell_type\": \"markdown\",\n   \"metadata\": {},\n   \"source\": [\n    \"# Exploration des Données Multi-Omiques\\n\",\n    \"\\n\",\n    \"Ce notebook présente l'analyse exploratoire des données.\"\n   ]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"execution_count\": null,\n   \"metadata\": {},\n   \"source\": [\n    \"import pandas as pd\\n\",\n    \"import numpy as np\\n\",\n    \"import matplotlib.pyplot as plt\\n\",\n    \"import seaborn as sns\\n\",\n    \"\\n\",\n    \"# Configuration\\n\",\n    \"plt.style.use('seaborn-v0_8')\\n\",\n    \"sns.set_palette(\\\"husl\\\")\\n\",\n    \"\\n\",\n    \"print(\\\"Environnement configuré pour l'EDA\\\")\"\n   ]\n  }\n ],\n \"metadata\": {\n  \"kernelspec\": {\n   \"display_name\": \"Python 3\",\n   \"language\": \"python\",\n   \"name\": \"python3\"\n  }\n },\n \"nbformat\": 4,\n \"nbformat_minor\": 4\n}",

        "tests/test_pipeline.py": """import pytest
from src.pipeline import MultiOmicsPipeline

def test_pipeline_initialization():
    pipeline = MultiOmicsPipeline()
    assert pipeline.config['general']['project_name'] == 'multi_omics_pipeline'

def test_pipeline_run():
    pipeline = MultiOmicsPipeline()
    # TODO: Ajouter des tests plus complets
    assert True
"""
    }
    
    for filename, content in files_to_create.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Créé : {filename}")
    
    print("\\n" + "="*60)
    print("🎉 Structure du projet créée avec succès!")
    print("="*60)
    print("\\nProchaines étapes:")
    print("1. cd projet-multi-omiques")
    print("2. python -m venv omics_env")
    print("3. source omics_env/bin/activate  # Linux/Mac")
    print("4. pip install -r requirements.txt")
    print("5. Commencer à développer!")

if __name__ == "__main__":
    create_simple_structure()