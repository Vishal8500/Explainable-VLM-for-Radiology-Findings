import pandas as pd
import evaluate

# ===============================
# LOAD CSV
# ===============================
df = pd.read_csv(r"D:\CV PROJ\vlm_predictions.csv")

# ===============================
# CLEAN DATA (VERY IMPORTANT)
# ===============================
df = df.dropna()

df["Reference"] = df["Reference"].astype(str).str.strip()
df["Prediction"] = df["Prediction"].astype(str).str.strip()

df = df[(df["Reference"] != "") & (df["Prediction"] != "")]

references = df["Reference"].tolist()
predictions = df["Prediction"].tolist()

print("Total Valid Samples:", len(predictions))

# ===============================
# LOAD METRICS
# ===============================
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# ===============================
# COMPUTE ROUGE
# ===============================
print("\nComputing ROUGE...")
rouge_scores = rouge.compute(
    predictions=predictions,
    references=references
)

# =============================== 
# COMPUTE BLEU
# ===============================
print("Computing BLEU...")
bleu_score = bleu.compute(
    predictions=predictions,
    references=[[r] for r in references]
)

# ===============================
# COMPUTE BERTSCORE
# ===============================
print("Computing BERTScore...")
bert_score = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)

# ===============================
# EXACT MATCH ACCURACY
# ===============================
correct = 0
for p, r in zip(predictions, references):
    if p.lower().strip() == r.lower().strip():
        correct += 1

accuracy = correct / len(predictions)

# ===============================
# PRINT RESULTS
# ===============================
print("\n============================")
print("📊 FINAL RESULTS")
print("============================")

print("\nROUGE:")
for k, v in rouge_scores.items():
    print(f"{k}: {round(v, 4)}")

print("\nBLEU Score:", round(bleu_score["bleu"], 4))

print("\nBERTScore F1:", round(sum(bert_score["f1"]) / len(bert_score["f1"]), 4))

print("\nExact Match Accuracy:", round(accuracy * 100, 2), "%")