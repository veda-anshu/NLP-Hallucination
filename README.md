# Project 7: Do LLMs Know When They're Making Things Up?

Hallucination detection on TruthfulQA with Qwen2.5-3B-Instruct.

```
code/task1/task1.ipynb   Task 1: token log-prob / entropy baseline
code/task2/task2.ipynb   Task 2: SelfCheckGPT (BERTScore and NLI)
code/task3/task3.ipynb   Task 3: logistic regression on Task 1 + Task 2 scores
outputs/task1/hand_check_filled.csv   our 50 hand labels
requirements.txt
report.pdf
```

## Setup

We ran everything on Google Colab (T4 GPU for Tasks 1 and 2, CPU is enough for Task 3).

1. Copy this folder to Google Drive as `MyDrive/project7/`.
2. Open a notebook in Colab, choose *Runtime > Change runtime type > T4 GPU*, and run all cells. The first cell mounts Drive and installs `bert-score`.

To run outside Colab, install the pinned packages with `pip install -r requirements.txt`, remove the Drive cell, and run the notebooks from the project folder.

## Datasets

We were assigned TruthfulQA and only use that one.

* **TruthfulQA** (used): `load_dataset("truthfulqa/truthful_qa", "generation", split="validation")`, 817 questions. The notebook downloads it automatically. (`truthful_qa` is the old name of the same dataset.) https://huggingface.co/datasets/truthfulqa/truthful_qa
* **WikiBio GPT-3 Hallucination** (not used): `load_dataset("potsawee/wiki_bio_gpt3_hallucination")`, split `evaluation`, 238 passages. https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination
* **HaluEval QA** (not used): download `data/qa_data.json` from https://github.com/RUCAIBox/HaluEval (JSON Lines, 10,000 examples).

## Pretrained models (Hugging Face)

* `Qwen/Qwen2.5-3B-Instruct`: answers and samples (Tasks 1, 2)
* `roberta-large`: BERTScore, for the labels (Task 1) and SelfCheck-BERTScore (Task 2)
* `microsoft/deberta-large-mnli`: SelfCheck-NLI (Task 2)

## How to run

Run in order; each task reads the previous task's outputs from `outputs/`.

| Notebook | Time | Writes |
|---|---|---|
| `task1.ipynb` | ~20 min (T4) | `outputs/task1/` (`task1_features.csv`, results, PR curve, hand check) |
| `task2.ipynb` | ~40 min (T4) | `outputs/task2/` (`samples.jsonl`, `task2_features.csv`, results, paper comparison) |
| `task3.ipynb` | ~1 min (CPU) | `outputs/task3/` (`split.csv`, `task3_results.csv`, PR curve) |

Hand check: Task 1 writes `outputs/task1/hand_check.csv` (50 answers). Fill `my_label` (1 = matches a correct reference, 0 = wrong), save it as `hand_check_filled.csv` in the same folder, and run the cell again to get the agreement with the automatic labels.

Task 2 saves samples after every batch, so if Colab disconnects, running it again continues from where it stopped.

## Settings and seeds

* System prompt: "Answer the question in one short sentence." Max 64 new tokens.
* Task 1: greedy decoding, `do_sample=False, repetition_penalty=1.0` (Qwen's config sets 1.05 by default, so we override it).
* Task 2: 20 samples per question, `temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0` (plain sampling; Qwen's config would otherwise add top-k 20, top-p 0.8 and the repetition penalty).
* Seeds: `SEED = 42` for torch/numpy, the 50-answer hand-check sample, and the Task 3 80/20 split (`random_state=42`, stratified). Task 2 batch `b` uses `torch.manual_seed(42 + first question id in the batch)`. The Task 3 stability check uses splits with `random_state = 0..99`.
