import pandas as pd
import matplotlib.pyplot as plt
from data.text_messages import test_messages

df = pd.read_csv('sentiment_results.csv')
message_truth = list(map(lambda tm: tm[1], test_messages))

algorithm_accuracy = {}
algorithms = df['Algorithm'].unique()

for algorithm in algorithms:
    alg_df = df[df['Algorithm'] == algorithm].sort_values('Sentence ID')
    preds = alg_df['Sentiment'].tolist()
    correct = sum([pred == true for pred, true in zip(preds, message_truth)])
    accuracy = correct / len(message_truth)
    algorithm_accuracy[algorithm] = accuracy

alg_names = list(algorithm_accuracy.keys())
accuracies = [v * 100 for v in algorithm_accuracy.values()] 

plt.figure(figsize=(8,6))
bars = plt.bar(alg_names, accuracies, color="#3498db", alpha=0.8, edgecolor="black")

for bar, acc, name in zip(bars, accuracies, alg_names):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{acc:.2f}%", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.text(bar.get_x() + bar.get_width()/2, 2, name, 
             ha='center', va='bottom', fontsize=10, color="black", fontweight='bold', alpha=0.7, rotation=90)

plt.ylim(0, 110)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Algorithm", fontsize=12)
plt.title("Accuracy by Algorithm", fontsize=14, pad=15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
