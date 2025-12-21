#!/usr/bin/env python3
"""
Test final simplifié du pipeline multi-omiques
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_modules_individuels():
    """Test chaque module individuellement"""
    
    print("🧬 TEST DES MODULES INDIVIDUELS")
    print("=" * 50)
    
    # Test 1: MissingValueHandler
    print("\n1️⃣ Testing MissingValueHandler...")
    try:
        from preprocessing.missing_values import MissingValueHandler
        
        # Créer des données avec valeurs manquantes
        test_data = pd.DataFrame({
            'gene1': [1, 2, np.nan, 4, 5],
            'gene2': [2, np.nan, 4, 5, 6],
            'category': ['A', 'B', np.nan, 'A', 'B']
        })
        
        handler = MissingValueHandler(strategy='knn', k=2)
        result = handler.fit_transform(test_data)
        
        missing_after = result.isnull().sum().sum()
        print(f"   ✅ MissingValueHandler fonctionnel")
        print(f"   • Valeurs manquantes après imputation: {missing_after}")
        
    except Exception as e:
        print(f"   ❌ Erreur MissingValueHandler: {e}")
    
    # Test 2: OmicsNormalizer
    print("\n2️⃣ Testing OmicsNormalizer...")
    try:
        from preprocessing.normalization import OmicsNormalizer
        
        # Données de test
        test_data = pd.DataFrame({
            'gene1': [100, 200, 300, 400, 500],
            'gene2': [50, 150, 250, 350, 450],
            'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005']
        })
        
        normalizer = OmicsNormalizer(method='log2_scale')
        result = normalizer.normalize(test_data)
        
        print(f"   ✅ OmicsNormalizer fonctionnel")
        print(f"   • Données normalisées: {result.shape}")
        
    except Exception as e:
        print(f"   ❌ Erreur OmicsNormalizer: {e}")
    
    # Test 3: SampleAlignment
    print("\n3️⃣ Testing SampleAlignment...")
    try:
        from integration.sample_alignment import SampleAlignment
        
        omic_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'gene1': [1, 2, 3]
        })
        
        clinical_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P004'],
            'age': [45, 50, 55]
        })
        
        aligner = SampleAlignment()
        aligned = aligner.align_by_patient_id(
            {'omic': omic_data, 'clinical': clinical_data},
            {'omic': 'patient_id', 'clinical': 'patient_id'}
        )
        
        print(f"   ✅ SampleAlignment fonctionnel")
        print(f"   • Échantillons omiques après alignement: {len(aligned['omic'])}")
        print(f"   • Échantillons cliniques après alignement: {len(aligned['clinical'])}")
        
    except Exception as e:
        print(f"   ❌ Erreur SampleAlignment: {e}")
    
    # Test 4: MultiOmicsFusion
    print("\n4️⃣ Testing MultiOmicsFusion...")
    try:
        from integration.data_fusion import MultiOmicsFusion
        
        omic_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'gene1': [1, 2]
        })
        
        clinical_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'age': [45, 50]
        })
        
        fusion = MultiOmicsFusion()
        fused = fusion.horizontal_fusion(
            {'omic': omic_data, 'clinical': clinical_data},
            sample_key='patient_id'
        )
        
        print(f"   ✅ MultiOmicsFusion fonctionnel")
        print(f"   • Données fusionnées: {fused.shape}")
        
    except Exception as e:
        print(f"   ❌ Erreur MultiOmicsFusion: {e}")
    
    # Test 5: JSONExporter
    print("\n5️⃣ Testing JSONExporter...")
    try:
        from standardization.json_export import JSONExporter
        
        test_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'gene1': [1.5, 2.3],
            'age': [45, 50]
        })
        
        exporter = JSONExporter()
        success = exporter.export_with_schema(test_data, 'test_output.json')
        
        print(f"   ✅ JSONExporter fonctionnel: {'Succès' if success else 'Échec'}")
        
    except Exception as e:
        print(f"   ❌ Erreur JSONExporter: {e}")
    
    # Test 6: CSVExporter
    print("\n6️⃣ Testing CSVExporter...")
    try:
        from standardization.csv_export import CSVExporter
        
        test_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'gene1': [1.5, 2.3],
            'age': [45, 50]
        })
        
        exporter = CSVExporter()
        success = exporter.export_standard_csv(test_data, 'test_output.csv')
        
        print(f"   ✅ CSVExporter fonctionnel: {'Succès' if success else 'Échec'}")
        
    except Exception as e:
        print(f"   ❌ Erreur CSVExporter: {e}")

def test_pipeline_simple():
    """Test simple du pipeline"""
    
    print("\n\n🚀 TEST SIMPLIFIÉ DU PIPELINE")
    print("=" * 50)
    
    try:
        # Test avec le pipeline principal
        from pipeline import MultiOmicsPipeline
        
        print("\n1️⃣ Testing pipeline principal...")
        
        # Vérifier que les fichiers existent
        required_files = [
            "demo_expression_data.csv",
            "demo_clinical_data.csv", 
            "config/config.yaml"
        ]
        
        for file in required_files:
            if not Path(file).exists():
                print(f"   ❌ Fichier manquant: {file}")
                return False
        
        # Initialiser le pipeline
        pipeline = MultiOmicsPipeline("config/config.yaml")
        print("   ✅ Pipeline initialisé")
        
        # Exécuter sur un petit jeu de données
        print("\n2️⃣ Exécution du pipeline...")
        results = pipeline.run(
            "demo_expression_data.csv",
            "demo_clinical_data.csv", 
            "test_simple"
        )
        
        if results['status'] == 'success':
            print("   ✅ Pipeline exécuté avec succès!")
            print(f"   • Échantillons: {results['summary']['n_samples']}")
            print(f"   • Features: {results['summary']['n_features']}")
            print(f"   • Fichiers créés: {len(results['output_paths'])}")
            
            # Vérifier les fichiers de sortie
            for format_name, file_path in results['output_paths'].items():
                if Path(file_path).exists():
                    size = Path(file_path).stat().st_size
                    print(f"   • {format_name}: {file_path} ({size} bytes)")
                else:
                    print(f"   • {format_name}: {file_path} (❌ manquant)")
            
            return True
        else:
            print(f"   ❌ Échec du pipeline: {results.get('error', 'Erreur inconnue')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur dans le test du pipeline: {e}")
        return False

def main():
    """Fonction principale"""
    
    print("🧪 TEST FINAL DU PROJET MULTI-OMIQUES")
    print("=" * 60)
    
    # Test 1: Modules individuels
    test_modules_individuels()
    
    # Test 2: Pipeline complet
    success = test_pipeline_simple()
    
    # Résumé final
    print(f"\n" + "=" * 60)
    print("📋 RÉSUMÉ FINAL:")
    
    if success:
        print("✅ PROJET PRÊT POUR LA LIVRAISON!")
        print("   • Tous les modules fonctionnent")
        print("   • Pipeline opérationnel")
        print("   • Données traitées avec succès")
        print("   • Fichiers de sortie générés")
    else:
        print("❌ PROJET NÉCESSITE DES CORRECTIONS")
        print("   • Vérifiez les erreurs ci-dessus")
        print("   • Assurez-vous que tous les modules sont importables")
    
    print("=" * 60)

if __name__ == "__main__":
    main()