# Somali Name Classifier

A binary classifier that distinguishes Somali names from non-Somali names, trained using LLaMA 3.1 8B Instruct with LoRA fine-tuning.

## Repository Structure

```
├── README.md
├── train_llama.py      # LLaMA 3.1 training script (Google Colab)
├── dupefinder.py       # Remove duplicate names from dataset
└── synthmaker.py       # Generate synthetic name pairs for augmentation
```

---

## Scripts

### `train_llama.py` — Model Training

Trains a LLaMA 3.1 8B Instruct binary classifier with LoRA. Designed to run on **Google Colab** with GPU.

**Prerequisites:**
- Google Colab with a GPU runtime
- HuggingFace account with access granted to `meta-llama/Llama-3.1-8B-Instruct`

**Setup:**
1. Upload `train_dataset.csv` and `eval_dataset.csv` to Google Drive at `/MyDrive/Datasets/`
2. Open `train_llama.py` in Colab and run all cells

**Dataset format:**
```csv
Full Name,label
AHMED MOHAMED,1
JOHN SMITH,0
FATIMA HUSSEIN,1
EMILY JOHNSON,0
```
- `label = 1` → Somali name
- `label = 0` → Non-Somali name

**Output:** Model and tokenizer saved to `/content/drive/MyDrive/Llama3_Binary_LoRA`

---

### `dupefinder.py` — Duplicate Cleaner

Scans `somali_classifier_full_dataset.csv` for duplicate names, reports a breakdown by label, and writes two output files.

```bash
python dupefinder.py
```

**Output files:**
- `duplicates_found.csv` — all duplicate rows
- `cleaned_dataset.csv` — deduplicated dataset ready for training

---

### `synthmaker.py` — Synthetic Name Generator

Generates synthetic name combinations from `gooddata.csv` by randomly mixing first and last name parts within each label group (Somali and non-Somali separately), then removes any names that already exist in the original dataset.

```bash
python synthmaker.py
```

**Input:** `gooddata.csv` with `Full Name` and `label` columns  
**Output:** `synthetic_dataset.csv`

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `v_proj` |
| LoRA dropout | 0.05 |
| Epochs | 3 |
| Batch size | 1 (gradient accumulation: 4) |
| Learning rate | 2e-4 |
| Precision | bfloat16 |

## Expected Performance

| Metric | Target |
|--------|--------|
| Accuracy | 85–95% |
| Dataset size | ~9,000–12,000 samples |
