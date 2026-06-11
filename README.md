# Evaluating the Compositional Generalization of Genomic Language Models

**Christos Jovanovic · Gavin Hearne · Michael Welch · Mokshad Sankhe**  
DSCI 691 · Natural Language Processing with Deep Learning · Drexel University

---

## Project Overview

This project investigates the use of pretrained genomic language models for promoter classification in human DNA sequences. We evaluate DNABERT2, a transformer model pretrained on large-scale genomic data, and compare its performance against several baseline approaches. The primary goal is to determine whether pretrained genomic representations improve generalization under distribution shifts introduced through motif-controlled dataset splits.

The project evaluates performance on in-distribution (ID) and out-of-distribution (OOD) datasets to assess both predictive accuracy and robustness. This is intended to replicate compositional generalization problems in NLP, applied to a genomic domain.

---

## Repository Structure

```
.
├── dataset_g/                    # Custom motif-aware splits
│   ├── dataset_pipeline.ipynb
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

## Dataset Construction

The data for this project is included under `/dataset_g`. If you intend on reproducing this data, you can run the `/dataset_g/data_pipeline.ipynb` notebook. This contains instructions as well as code used in downloading and processing the data for use in our models. This produces a final model-ready dataset, for use in downstream classifiers.

<p align="center">
  <img src="/Image/dataset_construction.png" alt="Dataset Construction Pipeline" width="400"/>
</p>

Producing this dataset was more challenging than initially expected, primarily because producing ID and OOD splits that were directly comparable required more careful construction than was done initially. After splitting the OOD test from the train dataset, we had initially produced a ID test through directly randomly sampling the training. This caused issues as a result of imbalanced motif frequencies. While both sets contained promoter and non-promoter sequences with the same motifs (only motif combinations were held-out), the OOD set was enriched for specific motifs. Sequences containing rare or highly distinct motifs were far easier to classify, so despite the OOD datsets containing novel motif combinations, the relatively higher volume of distinct motifs produced an easier-than-expected classification test. To account for this, we had to balance motif counts across both ID and OOD testing, as well as balance the class-labels. This made the two sets more directly comparable. 

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


The primary architecture for this project is the pretrained **DNABERT2-117M** genomic language model, sourced from Hugging Face at `zhihan1996/DNABERT-2-117M`. DNABERT2 is a transformer-based genomic language model built on the MosaicBERT architecture and trained on DNA sequences. For this project, DNABERT2 functions as a sequence encoder: it transforms tokenized DNA sequences into learned contextual representations.

For each input sequence, we pass the tokenized DNA through DNABERT2, mean-pool the resulting token embeddings, and feed the pooled representation into a custom classifier head. The final output is a binary prediction: promoter or non-promoter.

**Model pipeline:**

1. **DNA sequence input**  
    Each example is a fixed-length DNA sequence from the promoter classification dataset.

2. **Tokenization**  
    The raw DNA sequence is tokenized using the DNABERT2 tokenizer. Unlike fixed k-mer tokenization, DNABERT2 uses a BPE-style tokenizer with a learned fixed vocabulary, which represents sequences as variable-length DNA tokens.

3. **DNABERT2 encoder**  
    The tokenized sequence is passed through the pretrained DNABERT2 transformer encoder. The encoder produces contextual embeddings for each token, meaning each token representation depends on the surrounding sequence context.

4. **Sequence representation extraction**  
    The final hidden states from DNABERT2 are mean-pooled across the token dimension to produce a single sequence-level embedding. This gives one vector representation for the full DNA sequence.

5. **Classification head**  
    The pooled DNABERT2 embedding is passed through a custom classifier head:

    ```text
    pooled DNABERT2 embedding
    → LayerNorm
    → Dropout
    → Linear(hidden_size → hidden_size / 2)
    → GELU
    → Dropout
    → Linear(hidden_size / 2 → 2)
    ```

<p align="center">
  <img src="/Image/Architecture.png" alt="System Architecture Diagram" width="600"/>
</p>


**Training**

Training used class-weighted cross entropy to account for class imbalance. Models were optimized with **AdamW**, using a linear warmup/decay scheduler, gradient clipping, and early stopping based on validation performance.

**Training settings:**

- Optimizer: `AdamW`
- Loss: class-weighted cross entropy
- Batch size: `64`
- Early stopping patience: `5`
- Gradient clipping: `1.0`
- Scheduler: linear warmup / decay
- Learning rate:
    - LoRA fine-tuning: `4e-5`
    - Frozen encoder classifier training: `4e-4`
- LoRA settings:
    - Rank: `r = 8`
    - Alpha: `lora_alpha = 16`
    - Dropout: `lora_dropout = 0.1`
    - Target module: `Wqkv`

After training, each model was evaluated on the matched ID test set and the held-out motif-combination OOD test set.

There were several challenges faced relating to the development of the DNABERT2 classifier model, which were mostly a result of versioning. As a model that was developed a few years ago on different python and pytorch versions, we were faced with incompatibilities when we tried to run the model, both with smaller features and key functions. The largest issue was with the flashattention module Triton. This is designed as an optimization feature built into the DNABERT2 model, however trying to use it as intended resulted in an incompatibility with current versions of pytorch. As a result, we had to intentially uninstall it to make the model work as intended.

A second error was not catastrophic, but resulted in significant downstream performance losses. We pulled the model through `huggingface`, however had mistakenly loaded it through `BertModel.from_pretrained`. This was not intended by the creators of DNABERT2, and resulted in a number of weights being randomly initialized. Instead, we had to be careful to load it through `AutoModel.from_pretrained` to pull the correct weights and prevent nonstandard BERT layers from being overwritten. 

---

## Running the Project

**Setup:**

The bulk of this project was run through colab. There, the DNABERT2 notebook has been tested on python 3.12 with the following packages installed.

```bash
pip install transformers==4.38.2 tokenizers==0.15.2 accelerate==0.27.2 \
            peft==0.8.2 sentencepiece einops torch scikit-learn pandas tqdm
```

After installing, uninstall `triton` to avoid version conflicts. Without this uninstall, there may be conflicts with flash atention modules in the DNABERT2 architecture:

```bash
pip uninstall -y triton
```

**How to run:**
All notebooks are designed to be run seperately, however datasets need to be available prior to running any models. Datasets are available in this project, however users are encoraged to run and understand the dataset pipeline themselves, to better understand the dataset construction. GPU is highly encouraged, especially for the DNABERT2 model, which was primarily run on A100 GPUs available through colab. 

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
| K-mer + KNN | 0.753 | 0.739 |
| CNN | 0.913 | 0.737 |
| BiLSTM | 0.777 | 0.732 |

### DNABERT2 Models

| Model | ID F1 | OOD F1 |
|---|---|---|
| Frozen DNABERT2 | 0.899 | 0.827 |
| LoRA DNABERT2 | 0.910 | 0.847 |

---

## Evaluation Metrics

Models are evaluated using:

- F1 Score *(primary metric)*
- Balanced Accuracy
- ROC-AUC

---

## Challenges Encountered

- **Computational Requirements** — Training transformer-based genomic models requires significantly more memory and compute resources than traditional machine learning methods.
- **Model Adaptation** — Selecting an effective fine-tuning strategy required experimentation with frozen encoders and parameter-efficient methods such as LoRA.
- **Dataset Design** — Evaluating true biological generalization is challenging because models may memorize motif patterns rather than learn broader regulatory principles.

---

## Limitations

- Only one genomic foundation model (DNABERT2) was evaluated. There are a number of GLMs that have been produced: Nucleotide Transformer, HyenaDNA, etc... This also includes Protein transformers such as EVO, which operate on protein sequences instead of nucleotide sequences,
- Hyperparameter optimization was minimal.
- The study focuses only on promoter classification.
- OOD evaluation was limited to motif-based distribution shifts. Furthermore, we did only select a handful of motifs. This test could be expanded easily, working with different motifs, and different combinations of motifs than those that were selected in this project. 
- Biological validation of learned representations was outside the scope of this project. It would have been interesting to evaluate the hidden representations produced by the fine-tuned GLM. Comparing attention maps to the sequences has shown interesting results, highligting known biologically active sites in DNA in the past. 

---

## Future Work

- Evaluate additional genomic foundation models.
- Explore larger transformer architectures.
- Extend classification to other regulatory elements such as enhancers and splice sites.
- Investigate additional forms of biological distribution shift, such as taxanomic and functional.
- Apply interpretability methods to identify important genomic features learned by the model.
- Explore multi-task genomic prediction frameworks.

---

#### Contributor Profiles:
- **Christos Jovanovic** - 
- **Gavin Hearne** - [@ghproducts](https://github.com/ghproducts) 
- **Michael Welch** - 
- **Mokshad Sankhe** - [@Mokshu3242](https://github.com/Mokshu3242)
 
---

## Citations


**DNABERT2**

> Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R. V., & Liu, H. (2023).  
> **DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome.**  
> *arXiv:2306.15006*.  
> [https://arxiv.org/abs/2306.15006](https://arxiv.org/abs/2306.15006)

**MEME Suite**

> Bailey, T. L., Johnson, J., Grant, C. E., & Noble, W. S. (2015).  
> **The MEME Suite.**  
> *Nucleic Acids Research*, 43(W1), W39–W49.  
> [https://doi.org/10.1093/nar/gkv416](https://doi.org/10.1093/nar/gkv416)

**AME**

> McLeay, R. C., & Bailey, T. L. (2010).  
> **Motif Enrichment Analysis: a unified framework and an evaluation on ChIP data.**  
> *BMC Bioinformatics*, 11, 165.  
> [https://doi.org/10.1186/1471-2105-11-165](https://doi.org/10.1186/1471-2105-11-165)

**FIMO**

> Grant, C. E., Bailey, T. L., & Noble, W. S. (2011).  
> **FIMO: scanning for occurrences of a given motif.**  
> *Bioinformatics*, 27(7), 1017–1018.  
> [https://doi.org/10.1093/bioinformatics/btr064](https://doi.org/10.1093/bioinformatics/btr064)

**JASPAR**

> Ovek Baydar, D., Rauluseviciute, I., Aronsen, D. R., et al. (2026).  
> **JASPAR 2026: expansion of transcription factor binding profiles and integration of deep learning models.**  
> *Nucleic Acids Research*, 54(D1), D184–D193.  
> [https://doi.org/10.1093/nar/gkaf1209](https://doi.org/10.1093/nar/gkaf1209)

**Genomic Benchmarks**

> Grešová, K., Martinek, V., Čechák, D., Šimeček, P., & Alexiou, P. (2023).  
> **Genomic benchmarks: a collection of datasets for genomic sequence classification.**  
> *BMC Genomic Data*, 24, 25.  
> [https://doi.org/10.1186/s12863-023-01123-8](https://doi.org/10.1186/s12863-023-01123-8)


<!-- #DNABERT2:

#> Zhou et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. *arXiv:2306.15006* -->
