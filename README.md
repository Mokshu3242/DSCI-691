# Evaluating the Compositional Generalization of Genomic Language Models

**Christos Jovanovic · Gavin Hearne · Michael Welch · Mokshad Sankhe**  
DSCI 691 · Natural Language Processing with Deep Learning · Drexel University

---

## Project Overview

This project investigates the use of pretrained genomic language models for promoter classification in human DNA sequences. We evaluate DNABERT2, a transformer model pretrained on large-scale genomic data, and compare its performance against several baseline approaches. The primary goal is to determine whether pretrained genomic representations improve generalization under distribution shifts introduced through motif-controlled dataset splits.

The project evaluates performance on in-distribution (ID) and out-of-distribution (OOD) datasets to assess both predictive accuracy and robustness.

---

## Repository Structure

```
.
├── dataset_g/                    # Custom motif-aware splits
│   ├── train.csv
│   ├── test_ID.csv
│   ├── test_OOD.csv
│   └── test_motif_matched_ID.csv
│
├── dataset_m/                    # Pre-split dataset
│   └── promoter_final_dataset.csv
│
├── saved_models/                 # Model checkpoints
│
├── Baseline.ipynb                # KNN, CNN, BiLSTM
├── Baseline_Ablation.ipynb       # Ablation + extended analysis
├── DNABERT2.ipynb                # DNABERT2 (frozen + LoRA)
│
├── baseline_results.csv
└── README.md
```

---

## Architecture

<p align="center">
  <img src="/Image/Architecture.png" alt="System Architecture Diagram" width="600"/>
</p>


---

## Baseline Models

The following baseline models were evaluated:

| Model | Description |
|---|---|
| K-mer + KNN | 3-mer frequency vectors + K-Nearest Neighbors (K=15) |
| CNN | Two Conv1D layers over one-hot encoded sequence |
| BiLSTM | Single bidirectional LSTM with mean pooling |

These models provide a comparison against transformer-based genomic language models.

---

## DNABERT2 Model

DNABERT2 is a pretrained transformer architecture designed specifically for DNA sequences. Built on `zhihan1996/DNABERT-2-117M`, a classification head sits on top of mean-pooled encoder hidden states. The head is: LayerNorm, Dropout, Linear, GELU, Dropout, Linear.

**Pipeline:**

1. DNA sequence input
2. Tokenization
3. DNABERT2 encoder
4. Sequence representation extraction
5. Classification head
6. Promoter / Non-Promoter prediction

**Two training strategies were evaluated:**

- **Frozen Encoder** — the pretrained DNABERT2 encoder remains fixed while only the classification head is trained
- **LoRA Fine-Tuning** — Low-Rank Adaptation is applied to selected attention layers, allowing efficient task-specific adaptation while updating only a small subset of model parameters

---

## Running the Project

**Setup:**

```bash
pip install transformers==4.38.2 tokenizers==0.15.2 accelerate==0.27.2 \
            peft==0.8.2 sentencepiece einops torch scikit-learn pandas tqdm
```

After installing, uninstall `triton` to avoid version conflicts:

```bash
pip uninstall -y triton
```

**How to run:**

Run the notebooks in this order:

1. `Baseline.ipynb` — trains and evaluates KNN, CNN, BiLSTM on both datasets
2. `Baseline_Ablation.ipynb` — ablation study with extended analysis (calibration, saliency, UMAP)
3. `DNABERT2.ipynb` — trains and evaluates DNABERT2 with both strategies

---

## Models and Results
 
Trained model checkpoints and result files are available on Google Drive:
 
[Download models and results](https://drive.google.com/drive/folders/1fdiKafkQhYxPsI5R-UdUQOsZwB7xiwVj?usp=sharing)
 
---

## Results Summary

### Baseline Models

| Model | ID F1 | OOD F1 |
|---|---|---|
| K-mer + KNN | 0.684 | 0.765 |
| CNN | 0.748 | 0.598 |
| BiLSTM | 0.678 | 0.512 |

### DNABERT2 Models

| Model | ID F1 | OOD F1 |
|---|---|---|
| Frozen DNABERT2 | 0.813 | 0.818 |
| LoRA DNABERT2 | 0.835 | 0.842 |

---

## Evaluation Metrics

Models are evaluated using:

- F1 Score *(primary metric)*
- Accuracy
- Balanced Accuracy
- ROC-AUC

---

## Challenges Encountered

- **Computational Requirements** — Training transformer-based genomic models requires significantly more memory and compute resources than traditional machine learning methods.
- **Model Adaptation** — Selecting an effective fine-tuning strategy required experimentation with frozen encoders and parameter-efficient methods such as LoRA.
- **Dataset Design** — Evaluating true biological generalization is challenging because models may memorize motif patterns rather than learn broader regulatory principles.

---

## Limitations

- Only one genomic foundation model (DNABERT2) was evaluated.
- Hyperparameter optimization was limited by computational resources.
- The study focuses only on promoter classification.
- OOD evaluation was limited to motif-based distribution shifts.
- Biological validation of learned representations was outside the scope of this project.

---

## Future Work

- Evaluate additional genomic foundation models.
- Explore larger transformer architectures.
- Extend classification to other regulatory elements such as enhancers and silencers.
- Investigate additional forms of biological distribution shift.
- Apply interpretability methods to identify important genomic features learned by the model.
- Explore multi-task genomic prediction frameworks.

---

## Citation

DNABERT2:

> Zhou et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. *arXiv:2306.15006*