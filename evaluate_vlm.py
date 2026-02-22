import torch
import torch.nn as nn
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import timm
from torchvision import transforms
from tqdm import tqdm
import evaluate

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "razent/SciFive-base-Pubmed_PMC"
VIT_MODEL = "vit_tiny_patch16_224"
MODEL_PATH = "./vlm_model/vlm_epoch_3.pt"

DATASET_PATH = r"D:\CV PROJ\dataset\train"

MAX_LEN = 64

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# =====================================================
# MODEL DEFINITION
# =====================================================
class VisionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model(VIT_MODEL, pretrained=False)
        self.vit.head = nn.Identity()

        self.projection = nn.Linear(192,768)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def forward(self,x):
        tokens = self.vit.forward_features(x)
        cls_token = tokens[:,0]
        projected = self.projection(cls_token).unsqueeze(1)
        return projected

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading trained VLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = VisionLanguageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded.")

# =====================================================
# LOAD DATASET
# =====================================================
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

print("Test samples:", len(test_dataset))

# =====================================================
# METRICS
# =====================================================
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

predictions = []
references = []
findings_list = []

# =====================================================
# INFERENCE LOOP
# =====================================================
for sample in tqdm(test_dataset):

    image = transform(sample["image"]).unsqueeze(0).to(DEVICE)
    true_text = sample["impression"]

    with torch.no_grad():
        embeddings = model(image)

        decoder_input_ids = torch.ones((1,1), dtype=torch.long).to(DEVICE)

        outputs = model.lm.generate(
            inputs_embeds=embeddings,
            decoder_input_ids=decoder_input_ids,
            max_length=MAX_LEN,
            num_beams=4
        )

    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions.append(pred_text)
    references.append(true_text)
    findings_list.append(sample["findings"])

# =====================================================
# SAVE RESULTS CSV
# =====================================================
df = pd.DataFrame({
    "Findings": findings_list,
    "Reference": references,
    "Prediction": predictions
})

df.to_csv("vlm_predictions.csv", index=False)
print("✅ Predictions saved.")

# =====================================================
# COMPUTE METRICS
# =====================================================
print("\nComputing metrics...")

rouge_score = rouge.compute(predictions=predictions, references=references)
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")

print("\n========= FINAL RESULTS =========")
print("ROUGE:", rouge_score)
print("BLEU:", bleu_score["bleu"])
print("BERTScore F1:", sum(bert_score["f1"]) / len(bert_score["f1"]))