import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('sentiment_results.csv')

fig, ax = plt.subplots(figsize=(18, 12))

colors = {
    'POSITIVE': '#2ecc71', 
    'NEGATIVE': '#e74c3c',
    'NEUTRAL': '#f1c40f'
}

algorithms = df['Algorithm'].unique()
n_algorithms = len(algorithms)
sentences = sorted(df['Sentence ID'].unique())
bar_width = 1.2/n_algorithms  
group_spacing = 1.5  
inner_spacing = 0.2  

x = np.arange(len(sentences)) * (n_algorithms * (bar_width + inner_spacing) + group_spacing)

for i, sentence in enumerate(sentences):
    for idx, algorithm in enumerate(algorithms):
        data = df[(df['Algorithm'] == algorithm) & (df['Sentence ID'] == sentence)]
        if not data.empty:
            sentiment = data['Sentiment'].values[0]
            confidence = data['Confidence'].values[0]
            
            x_pos = x[i] + idx * (bar_width + inner_spacing)
            
            bar = ax.bar(
                x_pos, 
                confidence, 
                width=bar_width,
                color=colors[sentiment],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.8
            )
            
            ax.text(
                x_pos + bar_width/10,
                0.05,
                algorithm.upper(),
                rotation=90,
                fontsize=7,
                va='bottom',
                ha='center',
                color='black',
                alpha=0.9,
                fontweight='bold'
            )

ax.set_xticks(x + (n_algorithms-1)*(bar_width + inner_spacing)/2)
ax.set_xticklabels([f'Message {i+1}' for i in range(len(sentences))], fontsize=10)
ax.set_xlabel('Numer wiadomości', fontsize=12, labelpad=15)
ax.set_ylabel('Poziom pewności', fontsize=12, labelpad=15)
ax.set_title('Porównanie wyników algorytmów z podziałem na wiadomości i sentyment', pad=25, fontsize=14)
ax.set_ylim(0, 1.1)

# legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['POSITIVE'], label='Pozytywny', alpha=0.8),
    Patch(facecolor=colors['NEGATIVE'], label='Negatywny', alpha=0.8),
    Patch(facecolor=colors['NEUTRAL'], label='Neutralny', alpha=0.8)
]
ax.legend(handles=legend_elements, 
          loc='upper right', 
          fontsize=10, 
          title='Sentyment',
          title_fontsize=11)

ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.4)

for i in range(len(sentences)-1):
    ax.axvline(x[i] + n_algorithms*(bar_width + inner_spacing) + group_spacing/2, 
               color='gray', 
               linestyle=':', 
               linewidth=0.8)

plt.tight_layout()
plt.show()