# ============================================================
#  Llama 3.1 8B Instruct Binary Classifier with LoRA (Colab)
# ============================================================

!pip install -U -q transformers==4.44.2 peft==0.11.1 datasets==3.0.1 accelerate==0.33.0 bitsandbytes evaluate==0.4.2 huggingface_hub

import os, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
import evaluate

# 2️⃣ Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SAVE_DIR = "/content/drive/MyDrive/Llama3_Binary_LoRA"
DATA_PATH = "/content/drive/MyDrive/Datasets"

TRAIN_CSV = os.path.join(DATA_PATH, "train_dataset.csv")
EVAL_CSV  = os.path.join(DATA_PATH, "eval_dataset.csv")

# 3️⃣ Load Dataset
dataset = load_dataset("csv", data_files={"train": TRAIN_CSV, "eval": EVAL_CSV})

# 4️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocess — use "Full Name" instead of "text"
def preprocess(batch):
    return tokenizer(batch["Full Name"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 5️⃣ Model + LoRA setup
# 5️⃣ Model + LoRA setup
from transformers import AutoConfig

# Load config safely
config = AutoConfig.from_pretrained(MODEL_ID, num_labels=2, use_auth_token=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
    use_auth_token=True
)

# ✅ Padding & config fixes
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1
print("✅ Padding token set:", tokenizer.pad_token_id)

# LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 6️⃣ Metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }

# 7️⃣ Training setup
args = TrainingArguments(
    output_dir=SAVE_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    bf16=True,
    logging_dir=f"{SAVE_DIR}/logs",
    logging_steps=50,
    report_to="none"
)

# 8️⃣ Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# 9️⃣ Train
trainer.train()

# 🔟 Save model + tokenizer
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Model and tokenizer saved to {SAVE_DIR}")
