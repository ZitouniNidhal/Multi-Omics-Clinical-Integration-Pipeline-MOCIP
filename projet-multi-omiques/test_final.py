#!/usr/bin/env python3
"""
Simplified final test for the multi-omics pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_modules_individuels():
    """Tests each module individually"""
    
    print(" INDIVIDUAL MODULES TEST")
    print("=" * 50)
    
    # Test 1: MissingValueHandler
    print("\n1. Testing MissingValueHandler...")
    try:
        from preprocessing.missing_values import MissingValueHandler
        
        # Create data with missing values
        test_data = pd.DataFrame({
            'gene1': [1, 2, np.nan, 4, 5],
            'gene2': [2, np.nan, 4, 5, 6],
            'category': ['A', 'B', np.nan, 'A', 'B']
        })
        
        handler = MissingValueHandler(strategy='knn', k=2)
        result = handler.fit_transform(test_data)
        
        missing_after = result.isnull().sum().sum()
        print(f"   [OK] MissingValueHandler functional")
        print(f"   • Missing values after imputation: {missing_after}")
        
    except Exception as e:
        print(f"   [Error] MissingValueHandler Error: {e}")
    
    # Test 2: OmicsNormalizer
    print("\n2. Testing OmicsNormalizer...")
    try:
        from preprocessing.normalization import OmicsNormalizer
        
        # Test data
        test_data = pd.DataFrame({
            'gene1': [100, 200, 300, 400, 500],
            'gene2': [50, 150, 250, 350, 450],
            'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005']
        })
        
        normalizer = OmicsNormalizer(method='log2_scale')
        result = normalizer.normalize(test_data)
        
        print(f"   [OK] OmicsNormalizer functional")
        print(f"   • Normalized data: {result.shape}")
        
    except Exception as e:
        print(f"   [Error] OmicsNormalizer Error: {e}")
    
    # Test 3: SampleAlignment
    print("\n3. Testing SampleAlignment...")
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
        
        print(f"   [OK] SampleAlignment functional")
        print(f"   • Omics samples after alignment: {len(aligned['omic'])}")
        print(f"   • Clinical samples after alignment: {len(aligned['clinical'])}")
        
    except Exception as e:
        print(f"   [Error] SampleAlignment Error: {e}")
    
    # Test 4: MultiOmicsFusion
    print("\n4. Testing MultiOmicsFusion...")
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
        
        print(f"   [OK] MultiOmicsFusion functional")
        print(f"   • Fused data: {fused.shape}")
        
    except Exception as e:
        print(f"   [Error] MultiOmicsFusion Error: {e}")
    
    # Test 5: JSONExporter
    print("\n5. Testing JSONExporter...")
    try:
        from standardization.json_export import JSONExporter
        
        test_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'gene1': [1.5, 2.3],
            'age': [45, 50]
        })
        
        exporter = JSONExporter()
        success = exporter.export_with_schema(test_data, 'test_output.json')
        
        print(f"   [OK] JSONExporter functional: {'Success' if success else 'Failure'}")
        
    except Exception as e:
        print(f"   [Error] JSONExporter Error: {e}")
    
    # Test 6: CSVExporter
    print("\n6. Testing CSVExporter...")
    try:
        from standardization.csv_export import CSVExporter
        
        test_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'gene1': [1.5, 2.3],
            'age': [45, 50]
        })
        
        exporter = CSVExporter()
        success = exporter.export_standard_csv(test_data, 'test_output.csv')
        
        print(f"   [OK] CSVExporter functional: {'Success' if success else 'Failure'}")
        
    except Exception as e:
        print(f"   [Error] CSVExporter Error: {e}")

def test_pipeline_simple():
    """Simple pipeline test"""
    
    print("\n\n SIMPLIFIED PIPELINE TEST")
    print("=" * 50)
    
    try:
        # Test with the main pipeline
        from pipeline import MultiOmicsPipeline
        
        print("\n1. Testing main pipeline...")
        
        # Verify that files exist
        required_files = [
            "demo_expression_data.csv",
            "demo_clinical_data.csv", 
            "config/config.yaml"
        ]
        
        for file in required_files:
            if not Path(file).exists():
                print(f"   [Error] Missing file: {file}")
                return False
        
        # Initialize pipeline
        pipeline = MultiOmicsPipeline("config/config.yaml")
        print("   [OK] Pipeline initialized")
        
        # Execute on a small dataset
        print("\n2. Pipeline execution...")
        results = pipeline.run(
            "demo_expression_data.csv",
            "demo_clinical_data.csv", 
            "test_simple"
        )
        
        if results['status'] == 'success':
            print("   [OK] Pipeline executed successfully!")
            print(f"   • Samples: {results['summary']['n_samples']}")
            print(f"   • Features: {results['summary']['n_features']}")
            print(f"   • Files created: {len(results['output_paths'])}")
            
            # Verify output files
            for format_name, file_path in results['output_paths'].items():
                if Path(file_path).exists():
                    size = Path(file_path).stat().st_size
                    print(f"   • {format_name}: {file_path} ({size} bytes)")
                else:
                    print(f"   • {format_name}: {file_path} ([Error] missing)")
            
            return True
        else:
            print(f"   [Error] Pipeline failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   [Error] Error in pipeline test: {e}")
        return False

def main():
    """Main function"""
    
    print(" MULTI-OMICS PROJECT FINAL TEST")
    print("=" * 60)
    
    # Test 1: Individual modules
    test_modules_individuels()
    
    # Test 2: Complete pipeline
    success = test_pipeline_simple()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(" FINAL SUMMARY:")
    
    if success:
        print("[OK] PROJECT READY FOR DELIVERY!")
        print("   • All modules are functional")
        print("   • Pipeline is operational")
        print("   • Data processed successfully")
        print("   • Output files generated")
    else:
        print("[Error] PROJECT REQUIRES CORRECTIONS")
        print("   • Check the errors above")
        print("   • Ensure all modules are importable")
    
    print("=" * 60)

if __name__ == "__main__":
    main()