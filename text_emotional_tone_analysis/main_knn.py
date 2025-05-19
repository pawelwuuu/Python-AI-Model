from utils.tokenizer import simple_tokenizer, filter_stopwords
from models.knn_model import KNNClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.visualizations import generate_wordcloud
from utils.results import save_sentiment_analysis_results
from data.text_messages import test_messages
from data.train_data import train_messages
from collections import Counter

def load_data():
    train_texts = [msg for msg, label in train_messages]
    train_labels = [label for msg, label in train_messages]
    return train_texts, train_labels

def initialize_vectorizer():
    return TfidfVectorizer(
        tokenizer=lambda text: filter_stopwords(simple_tokenizer(text)),
        token_pattern=None,
        lowercase=True,
        max_features=1500,
        ngram_range=(1, 2)
    )

def analyze_messages(model, vectorizer, messages):
    results = []
    all_words = []
    
    for i, message in enumerate(messages, 1):
        message = message[0]
        tokens = filter_stopwords(simple_tokenizer(message))
        all_words.extend(tokens)
        
        X = vectorizer.transform([message])
        label, confidence = model.predict_sentiment(X)
        
        final_label = label if confidence >= 0.6 else "NEUTRAL"
        
        results.append({
            'sentence_id': i,
            'sentiment': final_label,
            'confidence': confidence
        })
        
        print(f"\n--- Message {i} ---")
        print(f"Text: {message[:100]}{'...' if len(message)>100 else ''}")
        print(f"Predicted: {final_label} ({confidence*100:.1f}%)")
    
    return results, all_words

def main():
    train_texts, train_labels = load_data()
    
    class_distribution = Counter(train_labels)
    print("\nClass distribution in training data:")
    for cls, count in class_distribution.items():
        print(f"{cls}: {count} samples")
    
    vectorizer = initialize_vectorizer()
    X = vectorizer.fit_transform(train_texts)
    
    model = KNNClassifier(n_neighbors=5)
    accuracy, report = model.train(X, train_labels)
    
    print(f"\nModel Accuracy: {accuracy*100:.1f}%")
    print("Classification Report:")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.2f}")
    
    results, all_words = analyze_messages(model, vectorizer, test_messages)
    
    save_sentiment_analysis_results(results, "KNN")
    generate_wordcloud(all_words)

if __name__ == "__main__":
    main()