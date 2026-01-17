"""FHIR (Fast Healthcare Interoperability Resources) exporter."""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import logging
from ..exceptions import ExportError, FHIRError

logger = logging.getLogger(__name__)


class FHIRExporter:
    """Export multi-omics data to FHIR format."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize FHIR exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.fhir_version = self.config.get('fhir_version', 'R4')
        self.resource_types = self.config.get('resource_types', ['Patient', 'Observation'])
        self.base_url = self.config.get('base_url', 'http://example.org/fhir')
        
        # Load FHIR mappings
        self.fhir_mappings = self._load_fhir_mappings()
        
        logger.info(f"Initialized FHIRExporter for FHIR {self.fhir_version}")
    
    def export(self, integrated_data: Dict[str, Any], output_file: str, 
               **kwargs) -> Dict[str, Any]:
        """
        Export integrated data to FHIR format.
        
        Args:
            integrated_data: Integrated multi-omics data
            output_file: Output file path
            **kwargs: Additional parameters
            
        Returns:
            Export results
        """
        logger.info(f"Exporting data to FHIR format: {output_file}")
        
        try:
            # Extract data
            main_data = integrated_data.get('integrated_data', pd.DataFrame())
            
            if main_data.empty:
                raise FHIRError("No data available for FHIR export")
            
            # Create FHIR bundle
            fhir_bundle = self._create_fhir_bundle(main_data, **kwargs)
            
            # Save to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fhir_bundle, f, indent=2, default=str)
            
            result = {
                'status': 'success',
                'output_file': str(output_path),
                'fhir_version': self.fhir_version,
                'resource_count': len(fhir_bundle.get('entry', [])),
                'bundle_type': fhir_bundle.get('type'),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"FHIR export complete: {result['resource_count']} resources")
            
            return result
            
        except Exception as e:
            logger.error(f"FHIR export failed: {e}")
            raise FHIRError(f"FHIR export failed: {e}")
    
    def _load_fhir_mappings(self) -> Dict[str, Any]:
        """Load FHIR mappings from configuration."""
        try:
            # Try to load from file
            mapping_file = Path("config/fhir_mappings.json")
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Use default mappings
                return self._get_default_fhir_mappings()
                
        except Exception as e:
            logger.warning(f"Failed to load FHIR mappings: {e}")
            return self._get_default_fhir_mappings()
    
    def _get_default_fhir_mappings(self) -> Dict[str, Any]:
        """Get default FHIR mappings."""
        return {
            "patient": {
                "clinical_data": {
                    "patient_id": "Patient.id",
                    "age": "Patient.birthDate",
                    "gender": "Patient.gender",
                    "ethnicity": "Patient.extension.ethnicity",
                    "race": "Patient.extension.race"
                }
            },
            "observation": {
                "gene_expression": {
                    "gene_symbol": "Observation.code.coding.display",
                    "expression_value": "Observation.valueQuantity.value",
                    "unit": "Observation.valueQuantity.unit",
                    "tissue_type": "Observation.bodySite.coding.display"
                },
                "clinical_lab": {
                    "test_name": "Observation.code.coding.display",
                    "result_value": "Observation.valueQuantity.value",
                    "reference_range": "Observation.referenceRange.low/high"
                }
            },
            "diagnostic_report": {
                "omics_analysis": {
                    "analysis_type": "DiagnosticReport.code.coding.display",
                    "result_summary": "DiagnosticReport.conclusion",
                    "genes_analyzed": "DiagnosticReport.result",
                    "platform": "DiagnosticReport.performer.display"
                }
            },
            "specimen": {
                "sample_info": {
                    "sample_id": "Specimen.id",
                    "sample_type": "Specimen.type.coding.display",
                    "collection_date": "Specimen.collection.collectedDateTime",
                    "tissue_type": "Specimen.bodySite.coding.display"
                }
            },
            "code_systems": {
                "gene_expression": "http://loinc.org/48002-0",
                "protein_expression": "http://loinc.org/33717-4",
                "metabolite_level": "http://loinc.org/33717-4",
                "cancer_stage": "http://snomed.info/sct/399707004",
                "survival_time": "http://loinc.org/80437-2"
            },
            "units": {
                "expression_tpm": "TPM",
                "expression_fpkm": "FPKM",
                "protein_intensity": "relative intensity",
                "metabolite_concentration": "umol/L",
                "survival_months": "months"
            }
        }
    
    def _create_fhir_bundle(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create FHIR bundle from data."""
        bundle = {
            "resourceType": "Bundle",
            "id": f"multi-omics-bundle-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "meta": {
                "lastUpdated": datetime.now().isoformat(),
                "profile": [f"http://hl7.org/fhir/{self.fhir_version}/StructureDefinition/Bundle"]
            },
            "type": "transaction",
            "entry": []
        }
        
        # Create resources for each row
        for idx, row in data.iterrows():
            resources = self._create_resources_from_row(row, idx, **kwargs)
            bundle["entry"].extend(resources)
        
        return bundle
    
    def _create_resources_from_row(self, row: pd.Series, row_index: int, **kwargs) -> List[Dict[str, Any]]:
        """Create FHIR resources from a data row."""
        resources = []
        
        # Extract patient information
        patient_resource = self._create_patient_resource(row, row_index)
        if patient_resource:
            resources.append({
                "fullUrl": f"{self.base_url}/Patient/patient-{row_index}",
                "resource": patient_resource,
                "request": {
                    "method": "POST",
                    "url": "Patient"
                }
            })
        
        # Create observations for omics data
        observation_resources = self._create_observation_resources(row, row_index)
        resources.extend(observation_resources)
        
        # Create diagnostic report
        diagnostic_resource = self._create_diagnostic_report(row, row_index)
        if diagnostic_resource:
            resources.append({
                "fullUrl": f"{self.base_url}/DiagnosticReport/report-{row_index}",
                "resource": diagnostic_resource,
                "request": {
                    "method": "POST",
                    "url": "DiagnosticReport"
                }
            })
        
        # Create specimen resource
        specimen_resource = self._create_specimen_resource(row, row_index)
        if specimen_resource:
            resources.append({
                "fullUrl": f"{self.base_url}/Specimen/specimen-{row_index}",
                "resource": specimen_resource,
                "request": {
                    "method": "POST",
                    "url": "Specimen"
                }
            })
        
        return resources
    
    def _create_patient_resource(self, row: pd.Series, row_index: int) -> Optional[Dict[str, Any]]:
        """Create Patient FHIR resource."""
        patient = {
            "resourceType": "Patient",
            "id": f"patient-{row_index}",
            "meta": {
                "profile": [f"http://hl7.org/fhir/{self.fhir_version}/StructureDefinition/Patient"]
            }
        }
        
        # Extract patient data from row
        mappings = self.fhir_mappings.get('patient', {}).get('clinical_data', {})
        
        # Patient ID
        if 'patient_id' in row and pd.notna(row['patient_id']):
            patient["id"] = str(row['patient_id'])
        
        # Gender
        if 'gender' in row and pd.notna(row['gender']):
            gender_value = str(row['gender']).lower()
            if gender_value in ['male', 'female', 'other', 'unknown']:
                patient["gender"] = gender_value
            else:
                # Map common values
                gender_map = {
                    'm': 'male',
                    'f': 'female',
                    'male': 'male',
                    'female': 'female',
                    'man': 'male',
                    'woman': 'female'
                }
                patient["gender"] = gender_map.get(gender_value, 'unknown')
        
        # Birth date (age-based calculation)
        if 'age' in row and pd.notna(row['age']):
            try:
                age = float(row['age'])
                # Calculate approximate birth year
                current_year = datetime.now().year
                birth_year = current_year - int(age)
                patient["birthDate"] = f"{birth_year}-01-01"
            except (ValueError, TypeError):
                logger.warning(f"Invalid age value: {row['age']}")
        
        # Extensions for ethnicity and race
        extensions = []
        
        if 'ethnicity' in row and pd.notna(row['ethnicity']):
            extensions.append({
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                "valueCodeableConcept": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v3-Ethnicity",
                        "display": str(row['ethnicity'])
                    }]
                }
            })
        
        if 'race' in row and pd.notna(row['race']):
            extensions.append({
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                "valueCodeableConcept": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v3-Race",
                        "display": str(row['race'])
                    }]
                }
            })
        
        if extensions:
            patient["extension"] = extensions
        
        # Add identifier if we have a patient ID
        if 'patient_id' in row and pd.notna(row['patient_id']):
            patient["identifier"] = [{
                "use": "usual",
                "system": "http://example.org/patient-id",
                "value": str(row['patient_id'])
            }]
        
        return patient if len(patient) > 2 else None  # Return None if only basic fields
    
    def _create_observation_resources(self, row: pd.Series, row_index: int) -> List[Dict[str, Any]]:
        """Create Observation FHIR resources."""
        observations = []
        
        # Gene expression observations
        expression_observations = self._create_expression_observations(row, row_index)
        observations.extend(expression_observations)
        
        # Clinical observations
        clinical_observations = self._create_clinical_observations(row, row_index)
        observations.extend(clinical_observations)
        
        return observations
    
    def _create_expression_observations(self, row: pd.Series, row_index: int) -> List[Dict[str, Any]]:
        """Create gene expression observations."""
        observations = []
        
        # Look for gene expression columns
        expression_columns = [col for col in row.index if 'expression' in col.lower() or '_expr' in col]
        
        for col in expression_columns:
            if pd.notna(row[col]):
                try:
                    value = float(row[col])
                    
                    # Extract gene symbol from column name
                    gene_symbol = col.replace('expression_', '').replace('_expr', '').replace('gene_expression_', '')
                    
                    observation = {
                        "resourceType": "Observation",
                        "id": f"expression-{gene_symbol}-{row_index}",
                        "meta": {
                            "profile": [f"http://hl7.org/fhir/{self.fhir_version}/StructureDefinition/Observation"]
                        },
                        "status": "final",
                        "category": [{
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "laboratory",
                                "display": "Laboratory"
                            }]
                        }],
                        "code": {
                            "coding": [{
                                "system": "http://loinc.org",
                                "code": "48002-0",
                                "display": "Gene expression"
                            }],
                            "text": f"Expression level of {gene_symbol}"
                        },
                        "subject": {
                            "reference": f"Patient/patient-{row_index}"
                        },
                        "valueQuantity": {
                            "value": value,
                            "unit": self.fhir_mappings.get('units', {}).get('expression_tpm', 'TPM'),
                            "system": "http://unitsofmeasure.org",
                            "code": self.fhir_mappings.get('units', {}).get('expression_tpm', 'TPM')
                        }
                    }
                    
                    # Add effective date if available
                    if 'collection_date' in row and pd.notna(row['collection_date']):
                        observation["effectiveDateTime"] = row['collection_date']
                    
                    observations.append({
                        "fullUrl": f"{self.base_url}/Observation/expression-{gene_symbol}-{row_index}",
                        "resource": observation,
                        "request": {
                            "method": "POST",
                            "url": "Observation"
                        }
                    })
                    
                except (ValueError, TypeError):
                    logger.warning(f"Invalid expression value: {row[col]}")
        
        return observations
    
    def _create_clinical_observations(self, row: pd.Series, row_index: int) -> List[Dict[str, Any]]:
        """Create clinical observation resources."""
        observations = []
        
        # Common clinical observations
        clinical_mappings = {
            'age': ('Age', 'years', 'http://loinc.org/30525-0'),
            'tumor_stage': ('Cancer stage', '', 'http://snomed.info/sct/399707004'),
            'survival_time': ('Survival time', 'months', 'http://loinc.org/80437-2'),
            'tumor_size': ('Tumor size', 'mm', 'http://loinc.org/33717-4'),
            'lymph_nodes': ('Lymph nodes involved', 'count', 'http://loinc.org/33717-4')
        }
        
        for clinical_field, (display, unit, code) in clinical_mappings.items():
            if clinical_field in row and pd.notna(row[clinical_field]):
                try:
                    value = float(row[clinical_field])
                    
                    observation = {
                        "resourceType": "Observation",
                        "id": f"clinical-{clinical_field}-{row_index}",
                        "meta": {
                            "profile": [f"http://hl7.org/fhir/{self.fhir_version}/StructureDefinition/Observation"]
                        },
                        "status": "final",
                        "category": [{
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "survey",
                                "display": "Survey"
                            }]
                        }],
                        "code": {
                            "coding": [{
                                "system": "http://loinc.org" if code.startswith('http://loinc.org') else code.split('/')[0] + '.info',
                                "code": code.split('/')[-1],
                                "display": display
                            }],
                            "text": display
                        },
                        "subject": {
                            "reference": f"Patient/patient-{row_index}"
                        },
                        "valueQuantity": {
                            "value": value,
                            "unit": unit,
                            "system": "http://unitsofmeasure.org" if unit else "",
                            "code": unit
                        }
                    }
                    
                    observations.append({
                        "fullUrl": f"{self.base_url}/Observation/clinical-{clinical_field}-{row_index}",
                        "resource": observation,
                        "request": {
                            "method": "POST",
                            "url": "Observation"
                        }
                    })
                    
                except (ValueError, TypeError):
                    # Handle non-numeric values
                    if pd.notna(row[clinical_field]):
                        observation = {
                            "resourceType": "Observation",
                            "id": f"clinical-{clinical_field}-{row_index}",
                            "meta": {
                                "profile": [f"http://hl7.org/fhir/{self.fhir_version}/StructureDefinition/Observation"]
                            },
                            "status": "final",
                            "category": [{
                                "coding": [{
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "survey",
                                    "display": "Survey"
                                }]
                            }],
                            "code": {
                                "coding": [{
                                    "system": "http://loinc.org" if code.startswith('http://loinc.org') else code.split('/')[0] + '.info',
                                    "code": code.split('/')[-1],
                                    "display": display
                                }],
                                "text": display
                            },
                            "subject": {
                                "reference": f"Patient/patient-{row_index}"
                            },
                            "valueString": str(row[clinical_field])
                        }
                        
                        observations.append({
                            "fullUrl": f"{self.base_url}/Observation/clinical-{clinical_field}-{row_index}",
                            "resource": observation,
                            "request": {
                                "method": "POST",
                                "url": "Observation"
                            }
                        })
        
        return observations
    
    def _create_diagnostic_report(self, row: pd.Series, row_index: int) -> Optional[Dict[str, Any]]:
        """Create DiagnosticReport FHIR resource."""
        # Only create if we have omics analysis data
        omics_columns = [col for col in row.index if any(omics in col.lower() for omics in ['gene', 'protein', 'metabol'])]
        
        if not omics_columns:
            return None
        
        diagnostic_report = {
            "resourceType": "DiagnosticReport",
            "id": f"omics-report-{row_index}",
            "meta": {
                "profile": [f"http://hl7.org/fhir/{self.fhir_version}/StructureDefinition/DiagnosticReport"]
            },
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "GE",
                    "display": "Genetics"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "33717-4",
                    "display": "Genomic analysis"
                }],
                "text": "Multi-omics analysis"
            },
            "subject": {
                "reference": f"Patient/patient-{row_index}"
            },
            "issued": datetime.now().isoformat(),
            "conclusion": "Multi-omics analysis completed",
            "result": []
        }
        
        # Add references to observations
        # This is simplified - in practice, you'd track actual observation references
        gene_expression_cols = [col for col in omics_columns if 'gene' in col.lower() and 'expression' in col.lower()]
        if gene_expression_cols:
            diagnostic_report["result"].append({
                "reference": f"Observation/expression-GENE-{row_index}"
            })
        
        return diagnostic_report if len(diagnostic_report.get("result", [])) > 0 else None
    
    def _create_specimen_resource(self, row: pd.Series, row_index: int) -> Optional[Dict[str, Any]]:
        """Create Specimen FHIR resource."""
        specimen = {
            "resourceType": "Specimen",
            "id": f"specimen-{row_index}",
            "meta": {
                "profile": [f"http://hl7.org/fhir/{self.fhir_version}/StructureDefinition/Specimen"]
            }
        }
        
        # Sample ID
        if 'sample_id' in row and pd.notna(row['sample_id']):
            specimen["id"] = str(row['sample_id'])
            specimen["identifier"] = [{
                "value": str(row['sample_id'])
            }]
        
        # Sample type
        if 'sample_type' in row and pd.notna(row['sample_type']):
            specimen["type"] = {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0487",
                    "display": str(row['sample_type'])
                }],
                "text": str(row['sample_type'])
            }
        
        # Tissue type
        if 'tissue_type' in row and pd.notna(row['tissue_type']):
            specimen["collection"] = {
                "bodySite": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "display": str(row['tissue_type'])
                    }],
                    "text": str(row['tissue_type'])
                }
            }
        
        # Collection date
        if 'collection_date' in row and pd.notna(row['collection_date']):
            specimen["collection"] = specimen.get("collection", {})
            specimen["collection"]["collectedDateTime"] = row['collection_date']
        
        # Subject reference
        specimen["subject"] = {
            "reference": f"Patient/patient-{row_index}"
        }
        
        return specimen if len(specimen) > 3 else None  # Return None if only basic fields
    
    def validate_fhir_bundle(self, fhir_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate FHIR bundle structure.
        
        Args:
            fhir_bundle: FHIR bundle to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'resource_counts': {},
            'structure_issues': []
        }
        
        # Check bundle structure
        required_bundle_fields = ['resourceType', 'type', 'entry']
        
        for field in required_bundle_fields:
            if field not in fhir_bundle:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Missing required bundle field: {field}")
        
        if fhir_bundle.get('resourceType') != 'Bundle':
            validation_results['valid'] = False
            validation_results['errors'].append("Bundle resourceType must be 'Bundle'")
        
        # Validate entries
        entries = fhir_bundle.get('entry', [])
        
        for i, entry in enumerate(entries):
            if 'resource' not in entry:
                validation_results['warnings'].append(f"Entry {i} missing resource")
                continue
            
            resource = entry['resource']
            resource_type = resource.get('resourceType')
            
            if not resource_type:
                validation_results['warnings'].append(f"Entry {i} missing resourceType")
                continue
            
            # Count resource types
            validation_results['resource_counts'][resource_type] = validation_results['resource_counts'].get(resource_type, 0) + 1
            
            # Validate specific resource types
            if resource_type == 'Patient':
                self._validate_patient_resource(resource, validation_results)
            elif resource_type == 'Observation':
                self._validate_observation_resource(resource, validation_results)
            elif resource_type == 'DiagnosticReport':
                self._validate_diagnostic_report_resource(resource, validation_results)
            elif resource_type == 'Specimen':
                self._validate_specimen_resource(resource, validation_results)
        
        return validation_results
    
    def _validate_patient_resource(self, patient: Dict[str, Any], validation_results: Dict[str, Any]):
        """Validate Patient resource."""
        # Check required fields
        if 'gender' in patient and patient['gender'] not in ['male', 'female', 'other', 'unknown']:
            validation_results['warnings'].append(f"Patient has invalid gender: {patient.get('gender')}")
        
        if 'birthDate' in patient:
            try:
                datetime.fromisoformat(patient['birthDate'].replace('Z', '+00:00'))
            except:
                validation_results['warnings'].append(f"Patient has invalid birthDate format: {patient.get('birthDate')}")
    
    def _validate_observation_resource(self, observation: Dict[str, Any], validation_results: Dict[str, Any]):
        """Validate Observation resource."""
        if 'status' not in observation:
            validation_results['warnings'].append("Observation missing status")
        
        if 'code' not in observation:
            validation_results['warnings'].append("Observation missing code")
        
        # Check value fields
        value_fields = ['valueQuantity', 'valueString', 'valueBoolean', 'valueInteger', 'valueDateTime']
        has_value = any(field in observation for field in value_fields)
        
        if not has_value and 'component' not in observation:
            validation_results['warnings'].append("Observation missing value")
    
    def _validate_diagnostic_report_resource(self, diagnostic_report: Dict[str, Any], validation_results: Dict[str, Any]):
        """Validate DiagnosticReport resource."""
        if 'status' not in diagnostic_report:
            validation_results['warnings'].append("DiagnosticReport missing status")
        
        if 'code' not in diagnostic_report:
            validation_results['warnings'].append("DiagnosticReport missing code")
    
    def _validate_specimen_resource(self, specimen: Dict[str, Any], validation_results: Dict[str, Any]):
        """Validate Specimen resource."""
        if 'subject' not in specimen:
            validation_results['warnings'].append("Specimen missing subject reference")
    
    def create_fhir_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Create a validation report for FHIR bundle."""
        report = []
        
        report.append("FHIR BUNDLE VALIDATION REPORT")
        report.append("=" * 40)
        report.append(f"Validation Status: {'VALID' if validation_results['valid'] else 'INVALID'}")
        report.append("")
        
        if validation_results['errors']:
            report.append("ERRORS:")
            for error in validation_results['errors']:
                report.append(f"  - {error}")
            report.append("")
        
        if validation_results['warnings']:
            report.append("WARNINGS:")
            for warning in validation_results['warnings']:
                report.append(f"  - {warning}")
            report.append("")
        
        if validation_results['resource_counts']:
            report.append("RESOURCE COUNTS:")
            for resource_type, count in validation_results['resource_counts'].items():
                report.append(f"  {resource_type}: {count}")
            report.append("")
        
        return "\n".join(report)