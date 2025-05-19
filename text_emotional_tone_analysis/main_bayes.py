# main_naive_bayes.py
from utils.tokenizer import simple_tokenizer, filter_stopwords
from models.naive_bayes_model import train_naive_bayes_model
from sklearn.feature_extraction.text import CountVectorizer
from utils.visualizations import generate_wordcloud
from utils.results import save_sentiment_analysis_results
from data.text_messages import test_messages
from data.train_data import train_messages  

train_texts = [message for message, label in train_messages]
train_labels = [label for message, label in train_messages]

vectorizer = CountVectorizer(
    tokenizer=lambda text: filter_stopwords(simple_tokenizer(text)),
    lowercase=True
)
X = vectorizer.fit_transform(train_texts)

model, accuracy, _ = train_naive_bayes_model(X, train_labels)
print(f"\nModel accuracy on test set: {accuracy*100:.1f}%")

all_words = []
results = []

for i, message in enumerate(test_messages, 1):
    message = message[0]

    tokens = simple_tokenizer(message)
    filtered_tokens = filter_stopwords(tokens)
    all_words.extend(filtered_tokens)
    
    X_message = vectorizer.transform([message])
    
    proba = model.predict_proba(X_message)[0]
    predicted_class_idx = proba.argmax()
    confidence = proba[predicted_class_idx]
    predicted_class = model.classes_[predicted_class_idx]
    
    if confidence >= 0.6:
        final_label = predicted_class
    else:
        final_label = "NEUTRAL"
    
    results.append({
        'sentence_id': i,
        'sentiment': final_label,
        'confidence': confidence
    })
    
    print(f"\n--- Message {i} ---")
    print(f"Text: {message}")
    print(f"Predicted sentiment: {final_label}")
    print(f"Confidence: {confidence*100:.1f}%")

save_sentiment_analysis_results(results, algorithm_name="NaiveBayes")

generate_wordcloud(all_words)