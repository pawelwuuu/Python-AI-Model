# models/sentiment_model.py
from transformers import pipeline

# Model do analizy sentymentu
def create_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

def analyze_sentiment(model, message):
    return model(message)[0]
