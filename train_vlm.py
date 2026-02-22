import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
import timm
from torchvision import transforms

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "razent/SciFive-base-Pubmed_PMC"
VIT_MODEL = "vit_tiny_patch16_224"

BATCH_SIZE = 1
LR = 3e-4
MAX_LEN = 64
TOTAL_EPOCHS = 3

OUTPUT_DIR = "./vlm_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================
dataset = load_from_disk(r"D:\CV PROJ\dataset\train")

dataset = dataset.train_test_split(test_size=0.2, seed=42)
val_test = dataset["test"].train_test_split(test_size=0.5, seed=42)

train_split = dataset["train"]
val_split = val_test["train"]

# Filter invalid samples
def filter_valid(example):
    return (
        isinstance(example["impression"], str)
        and len(example["impression"].strip()) > 0
    )

train_split = train_split.filter(filter_valid)
val_split = val_split.filter(filter_valid)

print("Train:", len(train_split))
print("Val:", len(val_split))

# =====================================================
# TRANSFORM
# =====================================================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =====================================================
# DATASET CLASS
# =====================================================
class VLM_Dataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = image_transform(sample["image"])

        text = sample["impression"]
        if not text or not isinstance(text, str):
            text = "No acute findings."

        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "image": image,
            "labels": tokens["input_ids"].squeeze(0)
        }

train_dataset = VLM_Dataset(train_split)
val_dataset = VLM_Dataset(val_split)

# =====================================================
# MODEL
# =====================================================
class VisionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model(VIT_MODEL, pretrained=True)
        self.vit.head = nn.Identity()

        # Freeze lower layers
        for name, param in self.vit.named_parameters():
            if "blocks.10" not in name and "blocks.11" not in name:
                param.requires_grad = False

        self.projection = nn.Linear(192, 768)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def forward(self, image, labels=None):
        features = self.vit(image)
        projected = self.projection(features).unsqueeze(1)

        outputs = self.lm(
            inputs_embeds=projected,
            labels=labels
        )
        return outputs

model = VisionLanguageModel().to(DEVICE)

# =====================================================
# COLLATOR
# =====================================================
def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"image": images, "labels": labels}

# =====================================================
# TRAIN LOOP (EPOCH-WISE RESUME SAFE)
# =====================================================
for epoch in range(1, TOTAL_EPOCHS + 1):

    print(f"\n========== EPOCH {epoch} ==========")

    # LOAD previous epoch weights automatically
    prev_path = f"{OUTPUT_DIR}/vlm_epoch_{epoch-1}.pt"

    if epoch > 1 and os.path.exists(prev_path):
        print(f"Loading previous weights: {prev_path}")
        model.load_state_dict(torch.load(prev_path))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        learning_rate=LR,
        fp16=True,
        logging_steps=100,
        save_strategy="no",     # IMPORTANT
        eval_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    trainer.train()

    # SAVE epoch weights
    save_path = f"{OUTPUT_DIR}/vlm_epoch_{epoch}.pt"
    torch.save(model.state_dict(), save_path)

    print(f"✅ Saved: {save_path}")

print("\n🎉 TRAINING COMPLETE")