import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych
df = pd.read_csv('sentiment_results.csv')

# Utworzenie siatki wykresów
fig, ax = plt.subplots(figsize=(18, 12))

# Konfiguracja kolorów i parametrów
colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'}
algorithms = df['Algorithm'].unique()
n_algorithms = len(algorithms)
sentences = sorted(df['Sentence ID'].unique())
bar_width = 0.8/n_algorithms  # Dynamiczna szerokość słupków
group_spacing = 0.5

# Główne pozycje grup na osi X
x = np.arange(len(sentences)) * (n_algorithms + group_spacing)

# Iteracja po zdaniach i algorytmach
for i, sentence in enumerate(sentences):
    for idx, algorithm in enumerate(algorithms):
        # Pobierz dane dla kombinacji zdanie-algorytm
        data = df[(df['Algorithm'] == algorithm) & (df['Sentence ID'] == sentence)]
        if not data.empty:
            sentiment = data['Sentiment'].values[0]
            confidence = data['Confidence'].values[0]
            
            # Oblicz pozycję słupka
            x_pos = x[i] + idx * bar_width
            
            # Rysowanie słupka
            bar = ax.bar(x_pos, 
                        confidence, 
                        width=bar_width*0.9, 
                        color=colors[sentiment],
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=0.5)
            
            # Dodanie etykiety algorytmu
            ax.text(x_pos + bar_width/2,  # Pozycja X
                    0.02,                 # Pozycja Y (na dole słupka)
                    algorithm.upper(),    # Tekst
                    rotation=90,           # Obrót 90 stopni
                    fontsize=8,            # Rozmiar czcionki
                    va='bottom',           # Wyrównanie pionowe
                    ha='center',           # Wyrównanie poziome
                    color='black')

# Konfiguracja osi i etykiet
ax.set_xticks(x + (n_algorithms-1)*bar_width/2)
ax.set_xticklabels([f'Message {i+1}' for i in range(len(sentences))], fontsize=10)
ax.set_xlabel('Numer wiadomości', fontsize=12, labelpad=15)
ax.set_ylabel('Poziom pewności', fontsize=12, labelpad=15)
ax.set_title('Porównanie wyników algorytmów z podziałem na wiadomości i sentyment', pad=25, fontsize=14)
ax.set_ylim(0, 1.1)

# Dodanie legendy dla sentymentów
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['POSITIVE'], label='Pozytywny', alpha=0.8),
    Patch(facecolor=colors['NEGATIVE'], label='Negatywny', alpha=0.8)
]
ax.legend(handles=legend_elements, 
          loc='upper right', 
          fontsize=10, 
          title='Sentyment',
          title_fontsize=11)

# Linie pomocnicze i siatka
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.4)

# Dodatkowe odstępy między grupami
for i in range(len(sentences)-1):
    ax.axvline(x[i] + n_algorithms*bar_width + group_spacing/2, 
               color='gray', 
               linestyle=':', 
               linewidth=0.8)

plt.tight_layout()
plt.show()