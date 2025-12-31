import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def map_model(row):
    if row['sub_dataset'] == 'ai.jsonl': return 'Baseline'
    elif 'gpt4' in row['sub_dataset']: return 'GPT-4'
    else: return 'Llama-3'

df = pd.read_csv('results/final_evaluation_results.csv')
df['Model'] = df.apply(map_model, axis=1)
order = ['Pair 1 (Free)', 'Pair 2 (SI)', 'Pair 3 (Medium)', 'Pair 3 (Hard)']
df['Experiment_Group'] = pd.Categorical(df['Experiment_Group'], categories=order, ordered=True)

sns.set_theme(style="whitegrid", font="Arial", context="paper")

plt.figure(figsize=(8, 5))

base = df[df['Experiment_Group'].isin(['Pair 1 (Free)', 'Pair 2 (SI)'])].copy()
line_df = pd.concat([base.assign(Model='GPT-4'), base.assign(Model='Llama-3'), 
                     df[df['Experiment_Group'].str.contains('Pair 3')]])
line_df = line_df.drop_duplicates(subset=['Experiment_Group', 'Model'])

sns.lineplot(data=line_df, x='Experiment_Group', y='AUC', hue='Model', marker='o', linewidth=2.5)
plt.title("Staircase Performance Decay (AUC)")
plt.savefig('results/performance_decay_line.png', dpi=300)

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='AUC', y='Brier', hue='Model', style='Experiment_Group', s=150)
plt.title("Correlation: Detection Power vs. Uncertainty")
plt.savefig('results/confidence_error_scatter.png', dpi=300)