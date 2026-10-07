# STCKLA

STCKLA is a semi-supervised transcriptomic clustering framework that combines scFoundation representations with LLM-derived biological priors.

The workflow contains two stages:

1. **LLM prior generation** using Qwen and curated biological knowledge.
2. **Prior-guided semi-supervised clustering** based on scFoundation.

## Code layout

The implementation is organized into smaller modules. The entry-point names,
command-line arguments, default hyperparameters and two-stage workflow remain
unchanged. Run the commands below from the STCKLA directory; the example data and
model paths are relative to the current working directory.

```text
STCKLA/
├── llm.py                    # Stage 1 entry point
├── train.py                  # Stage 2 entry point
├── base.py                   # Compatibility exports for shared helpers
├── load.py                   # Original scFoundation loader, unchanged
├── pretrainmodels/           # scFoundation model implementations
├── stckla_modules/
│   ├── __init__.py
│   ├── base/                 # Data utilities, model heads, losses and metrics
│   ├── llm/                  # Knowledge, Qwen calls and prior construction
│   ├── train/                # Training, prior guidance and inference
│   └── logging_utils.py      # Optional diagnostic output
├── requirements.txt
└── README.md
```

When copying or uploading the project, include the entire `stckla_modules/`
directory alongside the entry points. Copying only `base.py`, `llm.py` and
`train.py` is not sufficient. Keep `load.py` and `pretrainmodels/` as well.
The refactor introduces no additional runtime dependencies.

The main locations for reading or editing the implementation are:

| Purpose                                    | File                                       |
| ------------------------------------------ | ------------------------------------------ |
| LLM arguments and defaults                 | `stckla_modules/llm/cli.py`              |
| LLM workflow                               | `stckla_modules/llm/pipeline.py`         |
| Qwen requests and response parsing         | `stckla_modules/llm/ollama_client.py`    |
| Prior construction and export              | `stckla_modules/llm/context_prior.py`    |
| Training arguments and defaults (`Args`) | `stckla_modules/train/config.py`         |
| Training setup and workflow                | `stckla_modules/train/pipeline.py`       |
| Training loop and checkpoint selection     | `stckla_modules/train/training_loop.py`  |
| Prior losses and visible-gene injection    | `stckla_modules/train/prior_guidance.py` |
| Inference and evaluation                   | `stckla_modules/train/evaluation.py`     |

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



## Hardware

The code was tested on NVIDIA RTX 4090 GPUs (24 GB VRAM).
Other CUDA-compatible GPUs may also be used, depending on the model configuration and batch size.

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
python llm.py \
  --data_path data/BRCA_mRNA_top.csv \
  --label_path data/BRCA_label_num.csv \
  --gene_index_path data/OS_scRNA_gene_index.19264.tsv \
  --subtype_map_path data/BRCA_label_mapping.xlsx \
  --prior_root data/bio_prior_processed \
  --split_dir outputs/BRCA/split \
  --output_dir outputs/BRCA/qwen_prior \
  --dataset_name "GS-BRCA" \
  --dataset_key BRCA \
  --dataset_mode bulk \
  --seed 0
```

The generated prior used by Stage 2 is saved under:

```text
outputs/BRCA/qwen_prior/qwen_artifacts/qwen_prior_for_scfoundation.npz
```

## Stage 2: Train STCKLA

Run the semi-supervised clustering model with the generated LLM prior:

```bash
python train.py \
  --data_path data/BRCA_mRNA_top.csv \
  --label_path data/BRCA_label_num.csv \
  --gene_index_path data/OS_scRNA_gene_index.19264.tsv \
  --ckpt_path models/models.ckpt \
  --exposed_save_dir outputs/BRCA/split \
  --qwen_prior_npz outputs/BRCA/qwen_prior/qwen_artifacts/qwen_prior_for_scfoundation.npz \
  --local_model_dir outputs/BRCA/model \
  --infer_save_path outputs/BRCA/inference \
  --k_fixed 5 \
  --seed 0
```

To run on a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path data/BRCA_mRNA_top.csv \
  --label_path data/BRCA_label_num.csv \
  --gene_index_path data/OS_scRNA_gene_index.19264.tsv \
  --ckpt_path models/models.ckpt \
  --exposed_save_dir outputs/BRCA/split \
  --qwen_prior_npz outputs/BRCA/qwen_prior/qwen_artifacts/qwen_prior_for_scfoundation.npz \
  --local_model_dir outputs/BRCA/model \
  --infer_save_path outputs/BRCA/inference \
  --k_fixed 5 \
  --seed 0
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

The default training hyperparameters are defined in
`stckla_modules/train/config.py` (`Args`), and the LLM argument defaults are defined
in `stckla_modules/llm/cli.py` (`build_parser`). They can be overridden with the same
command-line arguments as before.

The default LLM mode remains `prior_only`: it exports biological priors and keeps
`qwen_guided_expression.csv` identical to the aligned expression before Qwen.
Stage 2 continues to use the original expression input together with the exported
prior NPZ. Optional expression enhancement and other branches retain their
original argument switches.

## Diagnostic output

Detailed diagnostic output is quiet by default. To restore the detailed messages
that were made optional during refactoring, set this before launching a command:

```bash
export STCKLA_VERBOSE=1
```

To return to the default output level:

```bash
unset STCKLA_VERBOSE
```

Training progress, loss values, clustering metrics, warnings and key output paths
remain visible. This setting controls the project's optional diagnostic output;
it does not control third-party library logging.

## Optional refactoring records

The following files document the refactor and are not required at runtime:

- `refactor_manifest.json`: records where original definitions were moved,
  original source checksums and diagnostic print changes.
- `validation_results.json`: records comparison checks, the local test environment
  and validation limitations.
- `拆分说明.md`: detailed explanation of the module layout and compatibility checks.

These records may be retained locally without uploading them to the training
server. The `stckla_modules/` directory contains executable code and is required.
