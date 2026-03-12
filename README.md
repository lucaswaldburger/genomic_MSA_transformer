# Genomic MSA Transformer

This repository contains the code for the **Genomic MSA Transformer** project. The project uses DNA sequences in a multiple sequence alignment transformer. These are trained in an unsupervised fashion. A classifier then uses the embeddings from the transformer to classify operons within genomes of various organisms.

## Paper

For more details on the project, please refer to our paper: [Learning Genome Architecture Using MSA Transformers](https://github.com/EmaadKhwaja/Genomic_MSA_Transformer/blob/main/paper/CS294_Final_Paper.pdf)

![Model Diagram](paper/images/Model%20Diagram.png)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

| File | Description |
|------|-------------|
| `dataloader.py` | `OperonLoader` dataset class -- loads operon JSONs, builds MSA tensors |
| `operon_transformer/` | Core model: axial MSA transformer and classifier |
| `train_operon_transformer.py` | Pretrain the unsupervised MSA transformer |
| `train_operon_transformer_classifier.py` | Train the operon classifier on pretrained embeddings |
| `evaluate_performance.py` | Evaluate classifier (ROC, AUC, confusion matrix, metrics) |
| `create_config.py` | Build `data.csv` from OperonHunter GFF/STRING data |
| `ablation/` | Ablation study analysis scripts and outputs |

## Usage

### 1. Prepare data

```bash
python create_config.py
```

### 2. Pretrain the MSA transformer

```bash
python train_operon_transformer.py --device cuda:0 --img-set bsubtilis --epochs 200
```

### 3. Train the classifier

```bash
python train_operon_transformer_classifier.py --device cuda:0 --img-set bsubtilis --epochs 200
```

### 4. Evaluate

```bash
python evaluate_performance.py --device cuda:0 --img-set bsubtilis
```

All scripts accept `--help` for a full list of configurable parameters.
