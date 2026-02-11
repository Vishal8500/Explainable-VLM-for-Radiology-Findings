import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "razent/SciFive-base-Pubmed_PMC"

MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 64
BATCH_SIZE = 2          # SAFE for 6GB VRAM
GRAD_ACCUM = 8          # Effective batch = 16
EPOCHS = 4
LR = 3e-4
OUTPUT_DIR = "./scifive_impression"

# -------------------------
# LOAD DATASET
# -------------------------
dataset = load_from_disk(r"D:\CV PROJ\dataset\train")

# Remove image column (language-only stage)
dataset = dataset.remove_columns(["image"])

# Filter empty / invalid rows
def filter_empty(example):
    return (
        isinstance(example["findings"], str)
        and isinstance(example["impression"], str)
        and len(example["findings"].strip()) > 0
        and len(example["impression"].strip()) > 0
    )

dataset = dataset.filter(filter_empty)

# Train / Validation split
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def safe_text(x):
    if x is None or not isinstance(x, str):
        return ""
    return x.strip()

def preprocess(batch):
    findings = [safe_text(x) for x in batch["findings"]]
    impressions = [safe_text(x) for x in batch["impression"]]

    inputs = tokenizer(
        findings,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        impressions,
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

# 🔥 TOKENIZE AND REMOVE RAW TEXT COLUMNS
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["findings", "impression"]
)

# -------------------------
# MODEL
# -------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Memory optimizations
model.gradient_checkpointing_enable()
torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# DATA COLLATOR
# -------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# -------------------------
# TRAINING ARGUMENTS (Transformers 5.x)
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False
)

# -------------------------
# TRAINER
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)

# -------------------------
# TRAIN
# -------------------------
trainer.train()

# -------------------------
# SAVE FINAL MODEL
# -------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training completed and model saved.")
