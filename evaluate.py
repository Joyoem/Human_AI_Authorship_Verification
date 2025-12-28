import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split

def load_jsonl(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line)['text'])
    return texts

def evaluate_folder(folder_path):
    human_file = os.path.join(folder_path, "human.jsonl")
    ai_files = [f for f in os.listdir(folder_path) if f.startswith("ai") and f.endswith(".jsonl")]
    
    results = []
    
    for ai_file in ai_files:
        ai_path = os.path.join(folder_path, ai_file)
        h_texts = load_jsonl(human_file)
        a_texts = load_jsonl(ai_path)
        
        X = h_texts + a_texts
        y = [0]*len(h_texts) + [1]*len(a_texts)
        
        # 1. TF-IDF Char n-gram 3-5
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
        X_vec = vectorizer.fit_transform(X)
        
        # 2. train
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        y_prob = clf.predict_proba(X_test)[:, 1]  # AI probability
        y_pred = clf.predict(X_test)
        
        # 3. metrix calculation
        results.append({
            "sub_dataset": ai_file,
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Brier": brier_score_loss(y_test, y_prob),
            "Accuracy": accuracy_score(y_test, y_pred)
        })
        
    return results

# loop
config = {
    "Pair 1 (Free)": "data/pair1_free/hc3_qa",
    "Pair 2 (SI)": "data/pair2_semantic_preserving/hc3_si",
    "Pair 3 (Medium)": "data/pair3_imitation/mixset_polish_sentence",
    "Pair 3 (Hard)": "data/pair3_imitation/mixset_polish_token"
}

all_stats = []
for label, path in config.items():
    if os.path.exists(path):
        res = evaluate_folder(path)
        for r in res:
            r['Experiment_Group'] = label
            all_stats.append(r)

# compare
df = pd.DataFrame(all_stats)
print(df.to_string())
df.to_csv("final_evaluation_results.csv")

#------------------------------------
# visulisation
#-----------------------------------
# AUC 
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.barplot(
    data=df,
    x="Experiment_Group",
    y="AUC",
    ci="sd"
)
plt.ylim(0.5, 1.0)
plt.title("Detection Difficulty Across Generation Regimes (AUC)")
plt.ylabel("ROC-AUC")
plt.xlabel("")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("auc_by_pair.png")
plt.show()

# brier score
plt.figure(figsize=(8,5))
sns.barplot(
    data=df,
    x="Experiment_Group",
    y="Brier",
    ci="sd"
)
plt.title("Probability Calibration Across Generation Regimes (Brier)")
plt.ylabel("Brier Score")
plt.xlabel("")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("brier_by_pair.png")
plt.show()

# compare llama vs gpt4
plt.figure(figsize=(8,5))
sns.barplot(
    data=df,
    x="sub_dataset",
    y="AUC",
    hue="Experiment_Group"
)
plt.title("Model-wise Detection Performance")
plt.ylabel("ROC-AUC")
plt.xlabel("")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("auc_by_model.png")
plt.show()