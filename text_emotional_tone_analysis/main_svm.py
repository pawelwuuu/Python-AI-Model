# main.py
from utils.tokenizer import simple_tokenizer, filter_stopwords
from models.svm_model import train_svm_model
from utils.visualizations import generate_wordcloud
from utils.results import save_sentiment_analysis_results
from sklearn.feature_extraction.text import TfidfVectorizer
from data.text_messages import test_messages

# Inicjalizacja wektoryzatora TF-IDF
tfidf_vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer, stop_words='english')

all_words = []
results = []

# Przetwarzanie wiadomości (tylko tokenizacja dla chmury słów)
for message in test_messages:
    tokens = simple_tokenizer(message)
    filtered_tokens = filter_stopwords(tokens)
    all_words.extend(filtered_tokens)

# Wektoryzacja tekstu z TF-IDF
X = tfidf_vectorizer.fit_transform(test_messages)

# Generowanie etykiet - SVM będzie je sam przewidywał
# (używam prostego podziału na połowę zbioru dla przykładu)
labels = ['POSITIVE' if i % 2 == 0 else 'NEGATIVE' for i in range(len(test_messages))]

# Trenowanie modelu SVM
svm_model, accuracy, proba = train_svm_model(X, labels)

# Przygotowanie wyników w istniejącym formacie
for i, (message, probabilities) in enumerate(zip(test_messages, proba), 1):
    classes = svm_model.classes_
    predicted_class = classes[probabilities.argmax()]
    svm_confidence = probabilities.max()
    
    results.append({
        'sentence_id': i,
        'sentiment': predicted_class,
        'confidence': float(svm_confidence)  # Konwersja numpy.float do float
    })

# Zapis wyników
save_sentiment_analysis_results(results, algorithm_name="SVM")

print(f"\nSVM Model Accuracy: {accuracy * 100:.2f}%")

# Generowanie chmury słów
generate_wordcloud(all_words)