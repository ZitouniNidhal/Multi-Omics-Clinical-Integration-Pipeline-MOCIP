# Pipeline d'Intégration Multi-Omiques

## 🧬 Description

Ce projet développe un pipeline modulaire pour l'intégration de données multi-omiques (transcriptomiques, génomiques) avec des données cliniques dans le domaine de la santé. Le pipeline gère le nettoyage, la normalisation, l'intégration et l'export vers des standards d'interopérabilité comme FHIR R4.

## 🎯 Objectifs

- **Collecte** : Identifier et télécharger des jeux de données biomédicaux publics (TCGA, GEO)
- **Nettoyage** : Gérer les valeurs manquantes, normaliser les données
- **Intégration** : Fusionner les données multi-omiques et cliniques
- **Standardisation** : Exporter vers FHIR R4, JSON schématisé, CSV standardisé
- **Qualité** : Pipeline testé, documenté et prêt pour l'IA

## 📁 Structure du Projet

```
projet-multi-omiques/
├── src/                           # Code source
│   ├── data_collection/          # Modules de collecte TCGA/GEO
│   ├── preprocessing/            # Nettoyage et normalisation
│   ├── integration/              # Fusion multi-modalités
│   ├── standardization/          # Export FHIR/JSON/CSV
│   ├── utils/                    # Utilitaires communs
│   └── pipeline.py               # Pipeline principal
├── data/                         # Données
│   ├── raw/                      # Données brutes
│   ├── processed/                # Données traitées
│   └── external/                 # Données externes
├── notebooks/                    # Notebooks Jupyter
├── tests/                        # Tests unitaires
├── config/                       # Configuration YAML
├── docs/                         # Documentation
└── logs/                         # Fichiers de log
```

## 🚀 Installation et Configuration

### Prérequis

- Python 3.8+
- 4GB RAM minimum
- 10GB d'espace disque

### Installation

```bash
# 1. Cloner le repository
git clone https://github.com/votreusername/projet-multi-omiques.git
cd projet-multi-omiques

# 2. Créer l'environnement virtuel
python -m venv omics_env

# 3. Activer l'environnement
source omics_env/bin/activate  # Linux/Mac
# ou omics_env\Scripts\activate  # Windows

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Vérifier l'installation
python -c "from src.pipeline import MultiOmicsPipeline; print('Installation réussie!')"
```

### Configuration

Modifier le fichier `config/config.yaml` selon vos besoins :

```yaml
# Configuration principale
general:
  project_name: "multi_omics_pipeline"
  version: "1.0.0"

# Paramètres de préprocessing
preprocessing:
  missing_values:
    strategy: "knn"  # ou "median", "mean"
    k: 5
  normalization:
    method: "tmm"    # ou "deseq2", "tpm"
```

## 📊 Utilisation

### Données de Démonstration

Des données de démonstration sont incluses pour tester le pipeline :

```bash
# Données d'expression génique
demo_expression_data.csv     # 10 patients × 5 gènes

# Données cliniques
demo_clinical_data.csv       # 10 patients × 5 variables
```

### Exécution du Pipeline

```bash
# Test avec les données de démonstration
python src/pipeline.py \
    --omic-data demo_expression_data.csv \
    --clinical-data demo_clinical_data.csv \
    --output-dir results

# Avec des fichiers personnalisés
python src/pipeline.py \
    --omic-data data/raw/expression_data.csv \
    --clinical-data data/raw/clinical_data.csv \
    --output-dir results
```

### Utilisation comme Module Python

```python
from src.pipeline import MultiOmicsPipeline

# Initialiser le pipeline
pipeline = MultiOmicsPipeline(config_path="config/config.yaml")

# Exécuter le pipeline
results = pipeline.run(
    omic_data_path="data/raw/expression.csv",
    clinical_data_path="data/raw/clinical.csv",
    output_dir="results"
)

# Vérifier les résultats
if results['status'] == 'success':
    print(f"✅ Pipeline terminé avec succès!")
    print(f"📁 Fichiers de sortie : {results['output_paths']}")
    print(f"📈 Résumé : {results['summary']}")
```

## 🔧 Modules Principaux

### 1. Data Collection (`src/data_collection/`)

- **TCGADataCollector** : Collecte des données TCGA via GDC API
- **GEODataCollector** : Collecte des données GEO via NCBI API

### 2. Preprocessing (`src/preprocessing/`)

- **MissingValueHandler** : Gestion des valeurs manquantes (KNN, médiane)
- **OmicsNormalizer** : Normalisation TMM, DESeq2, TPM
- **QualityControl** : Contrôle qualité et détection d'outliers

### 3. Integration (`src/integration/`)

- **SampleAlignment** : Alignement des échantillons par ID patient
- **MultiOmicsFusion** : Fusion horizontale des données multi-modalités

### 4. Standardization (`src/standardization/`)

- **FHIRExporter** : Export vers format FHIR R4
- **JSONExporter** : Export JSON avec schéma de validation

## 📈 Tests et Validation

### Tests Unitaires

```bash
# Exécuter tous les tests
pytest tests/

# Avec couverture
pytest --cov=src tests/

# Test spécifique
pytest tests/test_preprocessing/test_missing_values.py
```

### Validation des Données

Le pipeline inclut automatiquement :
- Validation de la qualité des données
- Vérification des formats d'export
- Tests de cohérence post-intégration

## 📊 Visualisations

Des notebooks Jupyter sont inclus pour l'exploration des données :

- `notebooks/01_data_exploration.ipynb` : Analyse exploratoire complète
- `notebooks/02_data_cleaning_demo.ipynb` : Démonstration du nettoyage
- `notebooks/03_data_integration_demo.ipynb` : Démonstration de l'intégration

## 🔬 Jeux de Données Recommandés

### Pour le développement

- **TCGA-BRCA** : Cancer du sein (1,221 échantillons)
- **GEO GSE96058** : Cancer du sein métastatique (563 échantillons)

### Pour la validation

- **ICGC** : International Cancer Genome Consortium
- **ArrayExpress** : Archive de données de puces à ADN

## 📤 Formats de Sortie

### FHIR R4

Export conforme au standard HL7 FHIR R4 :
- Ressources Patient, Observation, DiagnosticReport
- Validation du schéma FHIR
- Support des ontologies LOINC/HGNC

### JSON Schématisé

Format JSON avec schéma de validation :
- Métadonnées complètes
- Traçabilité des transformations
- Validation automatique

### CSV Standardisé

Format CSV avec conventions :
- Séparateur tabulation
- En-têtes standardisés
- Documentation des colonnes

## 🛠️ Développement

### Architecture

Le pipeline suit une architecture modulaire :

```python
# Exemple de module
def process_data(self, data, config):
    """Process data with configuration"""
    # Validation
    if not self.validate_input(data):
        raise ValueError("Invalid input data")
    
    # Traitement
    processed = self.apply_transformation(data, config)
    
    # Vérification
    if not self.validate_output(processed):
        raise ValueError("Invalid output data")
    
    return processed
```

### Ajout de Nouveaux Modules

1. Créer le module dans `src/nouveau_module/`
2. Implémenter l'interface standard
3. Ajouter les tests unitaires
4. Documenter l'utilisation
5. Mettre à jour la configuration

## 📚 Documentation

### Documentation Technique

- **API Reference** : Docstrings complètes
- **Architecture** : Diagrammes de flux
- **Standards** : Conformité FHIR R4

### Guides d'Utilisation

- **Guide de démarrage rapide** : 5 minutes pour démarrer
- **Tutoriels** : Exemples pas à pas
- **FAQ** : Questions fréquentes

## 🤝 Contribution

### Processus de Contribution

1. Fork le repository
2. Créer une branche (`feature/nouvelle-fonctionnalité`)
3. Commit les changements
4. Push vers la branche
5. Créer une Pull Request

### Standards de Code

- **Style** : PEP 8
- **Tests** : Couverture >90%
- **Documentation** : Docstrings obligatoires
- **Review** : Code review avant merge

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- **TCGA** : The Cancer Genome Atlas
- **GEO** : Gene Expression Omnibus
- **HL7 FHIR** : Standard d'interopérabilité
- **Python Community** : Écosystème scientifique

## 📞 Support

Pour toute question ou problème :

- **Documentation** : Voir `docs/`
- **Issues** : GitHub Issues
- **Email** : votre.email@example.com

---

**🧬 Pipeline Multi-Omiques - Intégration intelligente pour la médecine de précision**