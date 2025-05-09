# main.py
from utils.tokenizer import simple_tokenizer, filter_stopwords
from models.sentiment_model import create_sentiment_analyzer, analyze_sentiment
from models.knn_model import train_knn_model
from utils.visualizations import generate_wordcloud
from utils.results import save_sentiment_analysis_results  # Importujemy funkcję zapisującą wyniki
from sklearn.feature_extraction.text import CountVectorizer
from data.text_messages import test_messages

# Inicjalizacja modelu do analizy sentymentu
sentiment_analyzer = create_sentiment_analyzer()

all_words = []
labels = []

# Lista do przechowywania wyników do zapisu
results = []

# Przetwarzanie wiadomości i analiza sentymentu
for i, message in enumerate(test_messages, 1):
    # Tokenizacja i czyszczenie
    tokens = simple_tokenizer(message)
    filtered_tokens = filter_stopwords(tokens)
    all_words.extend(filtered_tokens)
    
    # Analiza sentymentu z użyciem modelu DistilBERT
    result = analyze_sentiment(sentiment_analyzer, message)
    confidence = result['score']
    label_map = {"POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE"}
    
    # Przypisanie etykiety na podstawie progu
    if confidence < 0.6:
        final_label = "NEUTRAL"
    else:
        final_label = label_map.get(result['label'], "UNKNOWN")
    
    # Drukowanie wyników dla każdego zdania
    print(f"--- Message {i} ---")
    print(f"Text: {message}")
    print(f"Sentiment: {final_label} (confidence: {confidence*100:.2f}%)")
    
    # Zbieranie wyników do późniejszego zapisu
    results.append({
        'sentence_id': i,
        'sentiment': final_label,
        'confidence': confidence
    })

    # Dodanie etykiety do listy, aby użyć jej w KNN
    labels.append(final_label)

# Zapis wyników do pliku CSV
save_sentiment_analysis_results(results, algorithm_name="KNN")

# Wektoryzacja tekstu (zamiana tekstu na cechy numeryczne)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(test_messages)

# Trenowanie modelu KNN
knn_model, accuracy = train_knn_model(X, labels, n_neighbors=3)

# Drukowanie dokładności modelu KNN
print(f"\nKNN Model Accuracy: {accuracy * 100:.2f}%")

# Generowanie chmury słów
generate_wordcloud(all_words)
