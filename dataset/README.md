# Large Files and Data Access

## Recommended download: Zenodo

Large data files and other reproduction resources are available through Zenodo:

**DOI:** [10.5281/zenodo.22976359](https://doi.org/10.5281/zenodo.22976359)

For convenience, we recommend using the Zenodo record to download large datasets, model files, and other large artifacts required for reproduction.

## Alternative download with Git LFS

Some large files in this repository are managed using **Git Large File Storage (Git LFS)**.

To obtain the complete repository, including files tracked by Git LFS, make sure Git LFS is installed on your system.

```bash
git lfs install
git clone https://github.com/AIDataLab/STCKLA-Textual-Knowledge-Constraints-for-Transcriptomic-Clustering-and-Cancer-Subtyping.git
cd STCKLA-Textual-Knowledge-Constraints-for-Transcriptomic-Clustering-and-Cancer-Subtyping
git lfs pull
```

If the repository has already been cloned, retrieve the large files with:

```bash
git lfs pull
```

Without Git LFS, some large files may appear only as small pointer files instead of the actual data.

## Notes

- Source code and small configuration files can be downloaded directly from GitHub.
- Large datasets, model files, and other large artifacts can be downloaded from Zenodo or retrieved through Git LFS.
- For reproducibility, please use the file versions associated with this repository and the corresponding Zenodo record.
