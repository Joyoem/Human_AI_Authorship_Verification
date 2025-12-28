# Authorship Attribution vs. LLM Style Translator

This project investigates the adversarial impact of Large Language Models (LLMs) acting as "style translators." We evaluate how effectively LLMs can erase or imitate unique human writing styles across different generation regimes.

## Repository Structure

### Scripts
- **`extract_hc3.py`**: Separates the HC3 Plus dataset into `human.jsonl` and `ai.jsonl` based on ground-truth labels for QA and Semantic-Invariant (SI) tasks.
- **`extract_mixset_original_revised.py`**: A preprocessing script that extracts paired human-original and AI-revised texts from MixSet, categorizing them by model (Llama/GPT-4) and attack type.
- **`evaluate.py`**: The main evaluation pipeline. It extracts TF-IDF Char-ngram features, trains a Logistic Regression classifier, and computes metrics including AUC, F1-score, and Brier Score.

### Dataset Organization (`/data`)
- **`pair1_free`**: Contains "Free Generation" samples (HC3-QA/MixSet-Rewrite) where AI has minimal constraints, serving as the detection baseline.
- **`pair2_semantic_preserving`**: Contains "Semantic-Invariant" samples (HC3-SI), evaluating detection performance under task-specific constraints like translation or summarization.
- **`pair3_imitation`**: The **core experimental group** focusing on explicit stylistic imitation.
  - **Medium**: Sentence-level polishing.
  - **Hard**: Token-level polishing designed to disrupt stylometric fingerprints.

## Key Findings
Our results demonstrate a significant performance collapse in authorship verification as the attack granularity increases.
- **Detection Degradation**: AUC drops from ~0.99 (Free) to ~0.67 (Hard-GPT4), suggesting that fine-grained stylistic imitation makes AI text nearly indistinguishable from human writing in terms of character distribution.
- **Model Capability**: GPT-4 exhibits superior adversarial robustness compared to Llama in "Hard" imitation tasks.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Execute the evaluation: `python evaluate.py`
3. Results and visualizations will be generated in the `results/` folder.