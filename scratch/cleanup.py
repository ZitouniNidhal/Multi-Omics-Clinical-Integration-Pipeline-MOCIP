import os

path = 'multiomics_scitepress.tex'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Replace smart characters
replacements = {
    '\u2019': "'", # Smart single quote
    '\u2018': "'", # Smart single quote
    '\u201c': '"', # Smart double quote
    '\u201d': '"', # Smart double quote
    '\uff0c': ', ', # Chinese comma
    '\u2013': '--', # En-dash
    '\u2014': '---', # Em-dash
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Fix specific spacing/typos mentioned by user
content = content.replace('NumPy, Scikit-learn', 'NumPy, Scikit-learn') # Ensure no Chinese comma
content = content.replace("method='k-NN'", "method='k-NN'") # Ensure standard quotes

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleanup complete.")
