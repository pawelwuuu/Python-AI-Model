# main.py
from utils.tokenizer import simple_tokenizer, filter_stopwords
from models.svm_model import train_svm_model
from utils.visualizations import generate_wordcloud
from utils.results import save_sentiment_analysis_results
from sklearn.feature_extraction.text import TfidfVectorizer
from data.text_messages import test_messages

tfidf_vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer, stop_words='english')

all_words = []
results = []

for message in test_messages:
    message = message[0]
    tokens = simple_tokenizer(message)
    filtered_tokens = filter_stopwords(tokens)
    all_words.extend(filtered_tokens)

X = tfidf_vectorizer.fit_transform(list(map(lambda tm: tm[0], test_messages)))

labels = ['POSITIVE' if i % 2 == 0 else 'NEGATIVE' for i in range(len(test_messages))]

svm_model, accuracy, proba = train_svm_model(X, labels)

for i, (message, probabilities) in enumerate(zip(test_messages, proba), 1):
    message = message[0]
    classes = svm_model.classes_
    predicted_class = classes[probabilities.argmax()]
    svm_confidence = probabilities.max()
    
    results.append({
        'sentence_id': i,
        'sentiment': predicted_class,
        'confidence': float(svm_confidence)
    })

save_sentiment_analysis_results(results, algorithm_name="SVM")

print(f"\nSVM Model Accuracy: {accuracy * 100:.2f}%")

generate_wordcloud(all_words)