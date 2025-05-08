import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

# Prosta funkcja tokenizacji bez NLTK
def simple_tokenizer(text):
    return re.findall(r'\b\w{3,}\b', text.lower())  # Znajduje tylko słowa o długości >= 3

# Ręczna lista angielskich stopwords (skrócona wersja)
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'at'
}

# Model do analizy sentymentu
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Przykładowe wiadomości
test_messages = [
    "I absolutely love this app! It's fantastic and very intuitive.",
    "This is the worst service I've ever experienced. Total failure.",
    "The book was average, neither good nor bad.",
    "I highly recommend this online store. Fast shipping and great prices!",
    "The app works okay. It crashes sometimes."
]

all_words = []

for i, message in enumerate(test_messages, 1):
    # Tokenizacja i czyszczenie
    tokens = simple_tokenizer(message)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    all_words.extend(filtered_tokens)
    
    # Analiza sentymentu
    result = sentiment_analyzer(message)[0]
    
    # Mapowanie wyników
    label_map = {
        "POSITIVE": "POSITIVE",
        "NEGATIVE": "NEGATIVE"
    }
    
    # Kategoryzacja z progiem neutralnym
    confidence = result['score']
    if confidence < 0.6:
        final_label = "NEUTRAL"
    else:
        final_label = label_map.get(result['label'], "UNKNOWN")

    print(f"\n--- Message {i} ---")
    print(f"Text: {message}")
    print(f"Sentiment: {final_label} (confidence: {confidence*100:.1f}%)")

# Generowanie chmury słów
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