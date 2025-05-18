# main.py
from utils.tokenizer import simple_tokenizer, filter_stopwords
from models.sentiment_model import create_sentiment_analyzer, analyze_sentiment
from utils.visualizations import generate_wordcloud
from utils.results import save_sentiment_analysis_results
from data.text_messages import test_messages

all_words = []

# Inicjalizacja modelu do analizy sentymentu
sentiment_analyzer = create_sentiment_analyzer()

# Lista do przechowywania wyników do zapisu
results = []

# Przetwarzanie wiadomości
for i, message in enumerate(test_messages, 1):
    # Tokenizacja i czyszczenie
    tokens = simple_tokenizer(message)
    filtered_tokens = filter_stopwords(tokens)
    all_words.extend(filtered_tokens)
    
    # Analiza sentymentu
    result = analyze_sentiment(sentiment_analyzer, message)
    
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

    # Drukowanie wyników dla każdego zdania
    print(f"\n--- Message {i} ---")
    print(f"Text: {message}")
    print(f"Sentiment: {final_label} (confidence: {confidence*100:.1f}%)")

    # Zbieranie wyników do późniejszego zapisu
    results.append({
        'sentence_id': i,
        'sentiment': final_label,
        'confidence': confidence
    })

# Zapis wyników do pliku CSV
save_sentiment_analysis_results(results, algorithm_name="Neural")

# Generowanie chmury słów
generate_wordcloud(all_words)
