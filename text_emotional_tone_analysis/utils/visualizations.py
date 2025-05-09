# utils/visualizations.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordcloud(all_words):
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis'
    ).generate(' '.join(all_words))

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud from Sample Messages", fontsize=16)
    plt.show()
