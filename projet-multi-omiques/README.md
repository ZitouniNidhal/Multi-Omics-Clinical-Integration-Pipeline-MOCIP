# Multi-Omics Clinical Integration Pipeline

## Large Data Management

This repository contains data and results that may be very large. To avoid GitHub errors related to file size limits and push timeouts, follow these best practices:

- do not commit raw data files (`data/raw/...`) to the main repository
- ignore generated or output folders:
  - `data/`
  - `results/`
  - `logs/`
- use Git LFS for large files that need tracking

## Git LFS recommended

If you need to keep large files in history, use Git LFS:

```bash
git lfs install
git lfs track "*.txt"
git lfs track "*.csv"
git lfs track "*.seg"
git lfs track "*.png"
```

Then add and commit:

```bash
git add .gitattributes
git add <your-large-files>
git commit -m "Track large files with Git LFS"
```

## Quick recommendation

- Keep source code in `src/`
- Keep large data outside the repository or from external sources
- Do not version large files that exceed 100 MB
- Use `git status` before each push to verify the state
