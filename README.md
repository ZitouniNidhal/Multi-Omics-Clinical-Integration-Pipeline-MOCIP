# Quick Start Guide – Multi-Omics Pipeline

## 🚀 Installation and Usage 

### 1. Download

```bash
# Extract the archive
tar -xzf projet-multi-omiques-livraison-finale.tar.gz
cd projet-multi-omiques
```

### 2. Install Dependencies

```bash
# Install Python 3.8+ if needed
# python --version

# Install required dependencies
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml

# For tests and notebooks (optional)
pip install jupyter pytest
```

### 3. Quick Test

```bash
# Run the demonstration
python demo_simple.py

# You should see:
# ✅ Fully functional end-to-end pipeline
# ✅ Cleaned and integrated data
# ✅ Export to standard formats
```

---

## 📋 Basic Usage

### Full Pipeline

```python
# Run the pipeline on your own data
python src/pipeline.py \
    --omic-data your_expression_data.csv \
    --clinical-data your_clinical_data.csv \
    --output-dir results/
```

### Using Demo Data

```bash
# Use the included demo datasets
python src/pipeline.py \
    --omic-data demo_expression_data.csv \
    --clinical-data demo_clinical_data.csv \
    --output-dir demo_results/
```

---

## 🧬 Data Structure

### Expression Data (CSV)

```csv
patient_id,gene1,gene2,gene3,...
P001,100.5,200.3,150.2,...
P002,95.2,180.1,140.5,...
...
```

### Clinical Data (CSV)

```csv
patient_id,age,sex,stage,survival_months,treatment_response
P001,45,M,I,24,Responder
P002,50,F,II,18,Non-responder
...
```

---

## 📊 Results

### Output Files

* **`integrated_data.csv`** : Merged and cleaned dataset
* **`integrated_data.json`** : Data with metadata and schema
* **`pipeline.log`** : Complete execution log

### Result Quality

* ✅ **100% completeness**: No missing values
* ✅ **Normalized**: Standardized scale for analysis
* ✅ **Validated**: Consistency and integrity checks applied

---

## 🔧 Configuration

### Edit Parameters

Modify `config/config.yaml`:

```yaml
preprocessing:
  missing_values:
    strategy: "knn"  # or "median", "mean"
    k: 5
  
normalization:
  method: "log2_scale"  # or "tmm", "tpm", "zscore"

export:
  formats: ["csv", "json"]  # "fhir" optional
```

---

## 📚 Available Resources

### Documentation

* **`README.md`** : Full user guide
* **`RAPPORT_FINAL.md`** : Detailed technical documentation
* **`PLANNING_2_SEMAINES.md`** : Project timeline


### Data

* **`demo_expression_data.csv`** : Demo omics data (10×5)
* **`demo_clinical_data.csv`** : Demo clinical data (10×5)

---

## 🎯 Key Features

### ✅ Preprocessing

* **KNN Imputation**: Handles missing values
* **Normalization**: Log2 transformation + standardization
* **Quality validation**: Automatic issue detection

### ✅ Integration

* **Alignment**: By patient ID
* **Merging**: Horizontal concatenation
* **Scaling**: Optional feature scaling

### ✅ Standard Export

* **JSON**: Includes schema and metadata
* **CSV**: Standardized biomedical format
* **Compatibility**: FHIR-ready (extensible)

---

## 🔬 Real-World Data Usage

### Recommended Data Sources

* **TCGA**: The Cancer Genome Atlas
* **GEO**: Gene Expression Omnibus
* **ArrayExpress**: Microarray and sequencing archive

### Data Scale

* **Tested on**: 10 samples × 5 genes
* **Ready for**: 1000+ samples × 20,000+ genes
* **Memory requirement**: Minimum 4GB RAM

---

## 🛠️ Development

### Add New Features

```python
# In src/your_module/
class NewModule:
    def __init__(self, config):
        self.config = config
    
    def process(self, data):
        # Your logic here
        return processed_data
```

### Testing

```bash
# Test a specific module
python -m pytest tests/test_your_module.py

# Test the full pipeline
python test_final.py
```

---

## 📞 Support and Help

### Common Issues

1. **Import error**: Make sure you are in the correct directory
2. **Missing data**: Use the provided demo datasets
3. **Performance**: Optimized for medium-scale datasets

### Resources

* **Full documentation**: Available in the `/docs` folder
* **Examples**: Provided Jupyter notebooks
* **Tests**: Included validation scripts

---


---

*Quick Start Guide – Multi-Omics Project delivered on November 21, 2025*

