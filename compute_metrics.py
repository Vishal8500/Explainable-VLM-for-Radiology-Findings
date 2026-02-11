import evaluate
import pandas as pd

df = pd.read_csv("evaluation_results.csv")

predictions = df["Prediction"].tolist()
references = df["Reference"].tolist()

# ROUGE
rouge = evaluate.load("rouge")
rouge_scores = rouge.compute(predictions=predictions, references=references)

print("ROUGE Scores:")
print(rouge_scores)

# BLEU
bleu = evaluate.load("sacrebleu")
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])

print("\nBLEU Score:")
print(bleu_score)

# BERTScore (semantic similarity)
bertscore = evaluate.load("bertscore")
bert_results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)

print("\nBERTScore F1 Average:")
print(sum(bert_results["f1"]) / len(bert_results["f1"]))
