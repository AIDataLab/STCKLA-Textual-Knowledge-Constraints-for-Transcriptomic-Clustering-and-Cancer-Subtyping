# Large Files and Data Access

## Recommended download: Figshare

Some datasets and model files used in this project are too large for convenient distribution through a standard GitHub repository. To keep the GitHub repository lightweight and easy to clone, these large files are provided separately through Figshare.

**Figshare:** https://doi.org/10.6084/m9.figshare.34162653

We recommend downloading the large datasets, model files, and other reproduction resources directly from the Figshare record above.

## Alternative download with Git LFS

Some large files may also be tracked using **Git Large File Storage (Git LFS)**.

To obtain Git LFS-managed files, make sure Git LFS is installed and run:

```bash
git lfs install
git clone https://github.com/AIDataLab/STCKLA-Textual-Knowledge-Constraints-for-Transcriptomic-Clustering-and-Cancer-Subtyping.git
cd STCKLA-Textual-Knowledge-Constraints-for-Transcriptomic-Clustering-and-Cancer-Subtyping
git lfs pull
```

If the repository has already been cloned, simply run:

```bash
git lfs pull
```

Without Git LFS, some large files may appear only as small pointer files rather than the actual data.

## Notes

- Source code and small configuration files are provided directly through GitHub.
- Large datasets, model files, and other large artifacts are primarily provided through Figshare because of their file size.
- Git LFS provides an alternative way to retrieve files that are tracked in the GitHub repository.
- For reproducibility, please use the files associated with this repository and the corresponding Figshare record.
