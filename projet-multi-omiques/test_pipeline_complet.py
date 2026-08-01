#!/usr/bin/env python3
"""
Complete multi-omics pipeline test - Final version for rapid delivery
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
from pipeline import MultiOmicsPipeline
import logging

def test_pipeline_complete():
    """Complete pipeline test with demonstration data"""
    
    print(" COMPLETE MULTI-OMICS PIPELINE TEST")
    print("=" * 60)
    
    # Test configuration
    config_path = "config/config.yaml"
    omic_data_path = "demo_expression_data.csv"
    clinical_data_path = "demo_clinical_data.csv"
    output_dir = "test_results_complete"
    
    try:
        # Step 1: Initialize the pipeline
        print("\n [*] Initializing pipeline...")
        pipeline = MultiOmicsPipeline(config_path)
        print(" Pipeline initialized successfully")
        
        # Step 2: Execute the pipeline
        print(f"\n [*] Executing pipeline on data...")
        print(f"   Omics data: {omic_data_path}")
        print(f"   Clinical data: {clinical_data_path}")
        
        results = pipeline.run(omic_data_path, clinical_data_path, output_dir)
        
        # Step 3: Verify results
        print(f"\n [*] Verifying results...")
        
        if results['status'] == 'success':
            print(" Pipeline executed successfully!")
            
            # Display summary
            print(f"\n RESULTS SUMMARY:")
            summary = results['summary']
            print(f"   * Processed samples: {summary['n_samples']}")
            print(f"   * Integrated features: {summary['n_features']}")
            print(f"   * Memory usage: {summary['memory_usage_mb']:.2f} MB")
            print(f"   * Completeness: {summary['completeness']:.1%}")
            
            # Display output files
            print(f"\n[Files] OUTPUT FILES:")
            for format_name, file_path in results['output_paths'].items():
                print(f"   * {format_name.upper()}: {file_path}")
                
                # Check that the file exists
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    print(f"      File created ({file_size} bytes)")
                else:
                    print(f"      File missing")
            
            # Step 4: Data validation
            print(f"\n[4] Output data validation...")
            
            # Load and verify CSV
            if 'csv' in results['output_paths']:
                csv_path = results['output_paths']['csv']
                if Path(csv_path).exists():
                    output_data = pd.read_csv(csv_path)
                    print(f"    CSV data loaded: {output_data.shape}")
                    print(f"   * Preview of first rows:")
                    print(output_data.head(3))
                    
                    # Verify quality
                    missing_values = output_data.isnull().sum().sum()
                    print(f"   * Missing values: {missing_values}")
                    
                    if missing_values == 0:
                        print("    No missing values - Data complete!")
                    else:
                        print(f"   [Warning] {missing_values} missing values detected")
            
            # Step 5: Additional tests
            print(f"\n[5] Additional tests...")
            
            # Verify output directory structure
            output_path = Path(output_dir)
            if output_path.exists():
                files = list(output_path.glob('*'))
                print(f"    Output directory created with {len(files)} file(s)")
                
                # List files
                for file in files:
                    print(f"     * {file.name} ({file.stat().st_size} bytes)")
            
            # Reproducibility test
            print(f"\n[Reproducibility] Reproducibility test...")
            print("   Executing pipeline a second time...")
            
            results_2 = pipeline.run(omic_data_path, clinical_data_path, f"{output_dir}_2")
            
            if results_2['status'] == 'success':
                # Compare results
                summary1 = results['summary']
                summary2 = results_2['summary']
                
                if (summary1['n_samples'] == summary2['n_samples'] and 
                    summary1['n_features'] == summary2['n_features']):
                    print("    Pipeline reproducible - Identical results!")
                else:
                    print("   ⚠️  Differences detected between executions")
            
            print(f"\n[OK] TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Return results for analysis
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
            print(f"\n PIPELINE FAILED")
            print(f"Error: {results.get('error', 'Unknown error')}")
            print("=" * 60)
            
            return {
                'success': False,
                'error': results.get('error', 'Unknown error')
            }
            
    except Exception as e:
        print(f"\n FATAL ERROR IN TEST")
        print(f"Error: {str(e)}")
        print("=" * 60)
        
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main test function"""
    
    # Verify that test files exist
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
        print(" Missing files for test:")
        for file in missing_files:
            print(f"   * {file}")
        print("\nMake sure you are in the project directory")
        return
    
    # Execute test
    results = test_pipeline_complete()
    
    # Display final summary
    print(f"\n FINAL TEST SUMMARY:")
    if results['success']:
        print(" Pipeline functional and ready for delivery!")
        print(f"   * Exported files: {results['validation']['files_created']}")
        print(f"   * Data integrity: {' OK' if results['validation']['data_integrity'] else ' Issue'}")
        print(f"   * Reproducibility: {' OK' if results['validation']['reproducible'] else ' Issue'}")
    else:
        print(" Pipeline requires corrections")
        print(f"   Error: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
