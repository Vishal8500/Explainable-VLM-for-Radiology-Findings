import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

MODEL_PATH = "./scifive_impression"
MAX_INPUT_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# -------------------------
# Load Dataset
# -------------------------
dataset = load_from_disk(r"D:\CV PROJ\dataset\train")
dataset = dataset.remove_columns(["image"])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

predictions = []
references = []

# -------------------------
# Generate Predictions
# -------------------------
for sample in tqdm(test_dataset):
    findings = sample["findings"]
    impression = sample["impression"]

    inputs = tokenizer(
        findings,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions.append(pred)
    references.append(impression)

# Save predictions
df = pd.DataFrame({
    "Findings": test_dataset["findings"],
    "Reference": references,
    "Prediction": predictions
})

df.to_csv("evaluation_results.csv", index=False)

print("✅ Predictions saved to evaluation_results.csv")
