# STCKLA

STCKLA is a semi-supervised transcriptomic clustering framework that combines scFoundation representations with LLM-derived biological priors.

The workflow contains two stages:

1. **LLM prior generation** using Qwen and curated biological knowledge.
2. **Prior-guided semi-supervised clustering** based on scFoundation.

## Requirements

Recommended environment:

```text
Python 3.10.14
CUDA 12.1
PyTorch 2.1.0+cu121
```

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
numpy
pandas
scipy
scikit-learn
torch
tqdm
scanpy
anndata
h5py
openpyxl
einops
local-attention
```

Stage 1 additionally requires:

```text
Ollama
Qwen3.5-9B
```

For example:

```bash
ollama pull qwen3.5:9b
```

## Data preparation

A minimal BRCA example requires the following files:

```text
data/
├── BRCA_mRNA_top.csv
├── BRCA_label_num.csv
├── BRCA_label_mapping.xlsx
├── OS_scRNA_gene_index.19264.tsv
└── bio_prior_processed/
    └── qwen_rag/
        └── knowledge_chunks.jsonl

models/
└── models.ckpt
```

Where:

- `BRCA_mRNA_top.csv`: transcriptomic expression matrix.
- `BRCA_label_num.csv`: numeric subtype labels.
- `BRCA_label_mapping.xlsx`: mapping between numeric labels and subtype names.
- `OS_scRNA_gene_index.19264.tsv`: scFoundation gene index.
- `knowledge_chunks.jsonl`: biological knowledge collected from sources such as Reactome, TRRUST, GTEx and PanglaoDB.
- `models.ckpt`: pretrained scFoundation checkpoint.

The exposed/labeled subset does not need to be prepared manually. If no previous split is available, the code can generate and save the semi-supervised split according to the configured labeled ratio and random seed.

## Stage 1: Generate LLM biological priors

Start the Ollama service and make sure `qwen3.5:9b` is available.

```bash
export OLLAMA_HOST=127.0.0.1:21434
```

Then run:

```bash
python llm.py   --data_path data/BRCA_mRNA_top.csv   --label_path data/BRCA_label_num.csv   --gene_index_path data/OS_scRNA_gene_index.19264.tsv   --subtype_map_path data/BRCA_label_mapping.xlsx   --prior_root data/bio_prior_processed   --split_dir outputs/BRCA/split   --output_dir outputs/BRCA/qwen_prior   --dataset_name "GS-BRCA"   --dataset_key BRCA   --dataset_mode bulk   --seed 0
```

The generated prior used by Stage 2 is saved under:

```text
outputs/BRCA/qwen_prior/qwen_artifacts/qwen_prior_for_scfoundation.npz
```

## Stage 2: Train STCKLA

Run the semi-supervised clustering model with the generated LLM prior:

```bash
python train.py   --data_path data/BRCA_mRNA_top.csv   --label_path data/BRCA_label_num.csv   --gene_index_path data/OS_scRNA_gene_index.19264.tsv   --ckpt_path models/models.ckpt   --exposed_save_dir outputs/BRCA/split   --qwen_prior_npz outputs/BRCA/qwen_prior/qwen_artifacts/qwen_prior_for_scfoundation.npz   --local_model_dir outputs/BRCA/model   --infer_save_path outputs/BRCA/inference   --k_fixed 5   --seed 0
```

To run on a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py   --data_path data/BRCA_mRNA_top.csv   --label_path data/BRCA_label_num.csv   --gene_index_path data/OS_scRNA_gene_index.19264.tsv   --ckpt_path models/models.ckpt   --exposed_save_dir outputs/BRCA/split   --qwen_prior_npz outputs/BRCA/qwen_prior/qwen_artifacts/qwen_prior_for_scfoundation.npz   --local_model_dir outputs/BRCA/model   --infer_save_path outputs/BRCA/inference   --k_fixed 5   --seed 0
```

## Output

The main outputs include:

```text
outputs/BRCA/
├── split/
├── qwen_prior/
│   └── qwen_artifacts/
│       └── qwen_prior_for_scfoundation.npz
├── model/
└── inference/
```

- `split/`: semi-supervised exposed/labeled split.
- `qwen_prior/`: LLM-derived biological priors.
- `model/`: trained model checkpoints.
- `inference/`: embeddings and clustering evaluation results.

## Notes

The example above uses the BRCA dataset with five molecular subtypes. For other datasets, modify the dataset paths, label mapping, dataset metadata, and `k_fixed` accordingly.

The default training and prior-generation hyperparameters are defined in the source code and can be overridden through command-line arguments when needed.
