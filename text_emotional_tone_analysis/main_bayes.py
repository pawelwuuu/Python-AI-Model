# main.py
from utils.tokenizer import simple_tokenizer, filter_stopwords
from models.sentiment_model import create_sentiment_analyzer, analyze_sentiment
from models.naive_bayes_model import train_naive_bayes_model
from utils.visualizations import generate_wordcloud
from utils.results import save_sentiment_analysis_results
from sklearn.feature_extraction.text import CountVectorizer
from data.text_messages import test_messages

# Inicjalizacja modelu do analizy sentymentu (etykiety referencyjne)
sentiment_analyzer = create_sentiment_analyzer()

all_words = []
labels = []
results = []

# Przetwarzanie wiadomości (jak w main_knn.py)
for i, message in enumerate(test_messages, 1):
    tokens = simple_tokenizer(message)
    filtered_tokens = filter_stopwords(tokens)
    all_words.extend(filtered_tokens)
    
    result = analyze_sentiment(sentiment_analyzer, message)
    confidence = result['score']
    
    # Mapowanie etykiet z progiem neutralnym
    final_label = "NEUTRAL" if confidence < 0.6 else result['label']
    
    print(f"--- Message {i} ---")
    print(f"Sentiment: {final_label} (confidence: {confidence*100:.2f}%)")
    
    results.append({'sentence_id': i, 'sentiment': final_label, 'confidence': confidence})
    labels.append(final_label)

# Zapis wyników
save_sentiment_analysis_results(results, algorithm_name="NaiveBayes")

# Wektoryzacja tekstu
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(test_messages)

# Trenowanie modelu Naive Bayes
model, accuracy = train_naive_bayes_model(X, labels)
print(f"\nNaive Bayes Model Accuracy: {accuracy * 100:.2f}%")

generate_wordcloud(all_words)