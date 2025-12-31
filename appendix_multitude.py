import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# ==========================================
# path
# ==========================================
PATHS = {
    'main': 'data/multitude/multitude.csv',
    'attack_semantic': 'data/multitude/multitude_obfuscated_paraphrased-ChatGPT.csv',
    'attack_structural': 'data/multitude/multitude_obfuscated_backtranslated-m2m100-1.2B.csv',
    'attack_physical': 'data/multitude/multitude_obfuscated_HomoglyphAttack.csv'
}

def plot_results(results_df):
    print("\nStep 3: visualization...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7), dpi=300)
 
    ax = sns.barplot(
        data=results_df, 
        x='Attack', 
        y='AUC', 
        hue='Detector', 
        palette='viridis', 
        edgecolor='0.2'
    )
 
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10, fontweight='bold')
  
    plt.title('Robustness of Authorship Detectors Against Style Obfuscation', fontsize=15, pad=20, fontweight='bold')
    plt.xlabel('Adversarial Scenarios (MULTITuDE)', fontsize=12)
    plt.ylabel('Detection Performance (AUC)', fontsize=12)
    plt.ylim(0.5, 1.05)  
    plt.legend(title='Detector Type', loc='lower right', frameon=True)

    plt.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random Guess')
    
    plt.tight_layout()
    plt.savefig('detection_robustness_study.png')
    print("table saved to: detection_robustness_study.png")
    plt.show()

def run_hardcore_experiment():
    print("Step 1: training (MULTITuDE English Train Split)...")
    try:
        df_main = pd.read_csv(PATHS['main'], sep=',', header=0)
    except FileNotFoundError:
        print(f"path error  {PATHS['main']}")
        return

    train_en = df_main[(df_main['language'].astype(str).str.strip() == 'en') & 
                       (df_main['split'].astype(str).str.strip() == 'train')]
    
    if len(train_en) == 0:
        print("empty dataset")
        return

    detectors = {
        'Char_Detector (3-5 gram)': TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000),
        'Word_Detector (1-2 gram)': TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000)
    }
    
    models = {}
    for name, vec in detectors.items():
        X_train = vec.fit_transform(train_en['text'].fillna("").astype(str))
        model = LogisticRegression(solver='liblinear').fit(X_train, train_en['label'])
        models[name] = (vec, model)
        print(f" - {name} 训练完成。")

    # ---- Step 2: evluation ----
    all_results = []
    tasks = [
        ('Original', PATHS['main'], 'None'),
        ('GPT4_Paraphrase', PATHS['attack_semantic'], 'Semantic'),
        ('Back_Translate', PATHS['attack_structural'], 'Structural'),
        ('Homoglyph', PATHS['attack_physical'], 'Physical')
    ]

    print("\nStep 2: evaluating (Paired Obfuscation)...")
    for task_name, path, attack_type in tasks:
        try:
            df_task = pd.read_csv(path)
            df_task['language'] = df_task['language'].astype(str).str.strip()
            df_task_en = df_task[df_task['language'] == 'en']
            
            text_col = 'text' if task_name == 'Original' else 'generated'
            
            ai_texts = df_task_en[df_task_en['label'].astype(str) == '1'][text_col].fillna("").astype(str)
            human_texts = df_task_en[df_task_en['label'].astype(str) == '0'][text_col].fillna("").astype(str)

            if len(ai_texts) == 0 or len(human_texts) == 0:
                continue

            size = min(len(ai_texts), len(human_texts), 1000)
            ai_sample = ai_texts.sample(size, random_state=42)
            human_sample = human_texts.sample(size, random_state=42)
            
            y_true = [1]*size + [0]*size
            eval_texts = pd.concat([ai_sample, human_sample])
            
            for det_name, (vec, model) in models.items():
                X_test = vec.transform(eval_texts)
                probs = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_true, probs)
                all_results.append({
                    'Attack': task_name,
                    'Detector': det_name,
                    'AUC': round(auc, 4),
                    'Type': attack_type
                })
                print(f" complete {task_name} | {det_name}: AUC = {auc:.4f}")
        except Exception as e:
            print(f"   failed {task_name} : {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('final_obfuscation_results.csv', index=False)

    plot_results(results_df)
    
    print("\n--- summary ---")
    print(results_df)

if __name__ == "__main__":
    run_hardcore_experiment()