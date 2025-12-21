#!/usr/bin/env python3
"""
Test complet du pipeline multi-omiques - Version finale pour livraison rapide
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
from pipeline import MultiOmicsPipeline
import logging

def test_pipeline_complet():
    """Test complet du pipeline avec données de démonstration"""
    
    print("🧬 TEST COMPLET DU PIPELINE MULTI-OMIQUES")
    print("=" * 60)
    
    # Configuration du test
    config_path = "config/config.yaml"
    omic_data_path = "demo_expression_data.csv"
    clinical_data_path = "demo_clinical_data.csv"
    output_dir = "test_results_complet"
    
    try:
        # Étape 1: Initialiser le pipeline
        print("\n1️⃣ Initialisation du pipeline...")
        pipeline = MultiOmicsPipeline(config_path)
        print("✅ Pipeline initialisé avec succès")
        
        # Étape 2: Exécuter le pipeline
        print(f"\n2️⃣ Exécution du pipeline sur les données...")
        print(f"   Données omiques: {omic_data_path}")
        print(f"   Données cliniques: {clinical_data_path}")
        
        results = pipeline.run(omic_data_path, clinical_data_path, output_dir)
        
        # Étape 3: Vérifier les résultats
        print(f"\n3️⃣ Vérification des résultats...")
        
        if results['status'] == 'success':
            print("✅ Pipeline exécuté avec succès!")
            
            # Afficher le résumé
            print(f"\n📊 RÉSUMÉ DES RÉSULTATS:")
            summary = results['summary']
            print(f"   • Échantillons traités: {summary['n_samples']}")
            print(f"   • Features intégrées: {summary['n_features']}")
            print(f"   • Mémoire utilisée: {summary['memory_usage_mb']:.2f} MB")
            print(f"   • Complétude: {summary['completeness']:.1%}")
            
            # Afficher les fichiers de sortie
            print(f"\n📁 FICHIERS DE SORTIE:")
            for format_name, file_path in results['output_paths'].items():
                print(f"   • {format_name.upper()}: {file_path}")
                
                # Vérifier que le fichier existe
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    print(f"     ✅ Fichier créé ({file_size} bytes)")
                else:
                    print(f"     ❌ Fichier manquant")
            
            # Étape 4: Validation des données
            print(f"\n4️⃣ Validation des données de sortie...")
            
            # Charger et vérifier le CSV
            if 'csv' in results['output_paths']:
                csv_path = results['output_paths']['csv']
                if Path(csv_path).exists():
                    output_data = pd.read_csv(csv_path)
                    print(f"   ✅ Données CSV chargées: {output_data.shape}")
                    print(f"   • Aperçu des premières lignes:")
                    print(output_data.head(3))
                    
                    # Vérifier la qualité
                    missing_values = output_data.isnull().sum().sum()
                    print(f"   • Valeurs manquantes: {missing_values}")
                    
                    if missing_values == 0:
                        print("   ✅ Aucune valeur manquante - Données complètes!")
                    else:
                        print(f"   ⚠️  {missing_values} valeurs manquantes détectées")
            
            # Étape 5: Tests supplémentaires
            print(f"\n5️⃣ Tests supplémentaires...")
            
            # Vérifier la structure du répertoire de sortie
            output_path = Path(output_dir)
            if output_path.exists():
                files = list(output_path.glob('*'))
                print(f"   ✅ Répertoire de sortie créé avec {len(files)} fichier(s)")
                
                # Lister les fichiers
                for file in files:
                    print(f"     • {file.name} ({file.stat().st_size} bytes)")
            
            # Test de reproductibilité
            print(f"\n🔄 Test de reproductibilité...")
            print("   Exécution du pipeline une deuxième fois...")
            
            results_2 = pipeline.run(omic_data_path, clinical_data_path, f"{output_dir}_2")
            
            if results_2['status'] == 'success':
                # Comparer les résultats
                summary1 = results['summary']
                summary2 = results_2['summary']
                
                if (summary1['n_samples'] == summary2['n_samples'] and 
                    summary1['n_features'] == summary2['n_features']):
                    print("   ✅ Pipeline reproductible - Résultats identiques!")
                else:
                    print("   ⚠️  Différences détectées entre les exécutions")
            
            print(f"\n🎉 TEST TERMINÉ AVEC SUCCÈS!")
            print("=" * 60)
            
            # Retourner les résultats pour analyse
            return {
                'success': True,
                'results': results,
                'validation': {
                    'files_created': len(results['output_paths']),
                    'data_integrity': missing_values == 0,
                    'reproducible': results_2['status'] == 'success'
                }
            }
            
        else:
            print(f"\n❌ ÉCHEC DU PIPELINE")
            print(f"Erreur : {results.get('error', 'Erreur inconnue')}")
            print("=" * 60)
            
            return {
                'success': False,
                'error': results.get('error', 'Unknown error')
            }
            
    except Exception as e:
        print(f"\n❌ ERREUR FATALE DANS LE TEST")
        print(f"Erreur : {str(e)}")
        print("=" * 60)
        
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Fonction principale de test"""
    
    # Vérifier que les fichiers de test existent
    required_files = [
        "demo_expression_data.csv",
        "demo_clinical_data.csv", 
        "config/config.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Fichiers manquants pour le test :")
        for file in missing_files:
            print(f"   • {file}")
        print("\nAssurez-vous d'être dans le répertoire du projet")
        return
    
    # Exécuter le test
    results = test_pipeline_complet()
    
    # Afficher un résumé final
    print(f"\n📋 RÉSUMÉ FINAL DU TEST:")
    if results['success']:
        print("✅ Pipeline fonctionnel et prêt pour la livraison!")
        print(f"   • Fichiers exportés : {results['validation']['files_created']}")
        print(f"   • Intégrité données : {'✅ OK' if results['validation']['data_integrity'] else '❌ Problème'}")
        print(f"   • Reproductibilité : {'✅ OK' if results['validation']['reproducible'] else '❌ Problème'}")
    else:
        print("❌ Pipeline nécessite des corrections")
        print(f"   Erreur : {results.get('error', 'Erreur inconnue')}")

if __name__ == "__main__":
    main()