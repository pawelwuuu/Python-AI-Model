import pandas as pd
import matplotlib.pyplot as plt

# Załaduj dane
df = pd.read_csv('sentiment_results.csv')

# Oblicz średnie wartości Confidence dla każdej kombinacji Algorithm-Sentiment
pivot_df = df.groupby(['Algorithm', 'Sentiment'])['Confidence'].mean().unstack(0)

# Utwórz wykres
plt.figure(figsize=(12, 7))
ax = pivot_df.plot(kind='bar', width=0.8, edgecolor='black', alpha=0.9)

# Konfiguracja wyglądu
plt.title('Porównanie średniej pewności dla analizy sentymentu', fontsize=14, pad=20)
plt.xlabel('Sentyment', fontsize=12)
plt.ylabel('Średnia pewność', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0.7, 1.05)  # Zwiększ czytelność skali

# Dodaj etykiety wartości
for p in ax.containers:
    ax.bar_label(p, fmt='%.3f', label_type='edge', padding=2, fontsize=8)

# Popraw układ legendy
plt.legend(title='Algorytm', bbox_to_anchor=(1.05, 1), loc='upper left')

# Dodatkowe usprawnienia wizualne
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()