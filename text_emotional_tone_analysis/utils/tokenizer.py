# utils/tokenizer.py
import re

# Prosta funkcja tokenizacji bez NLTK
def simple_tokenizer(text):
    return re.findall(r'\b\w{3,}\b', text.lower())  # Znajduje tylko słowa o długości >= 3

# Ręczna lista angielskich stopwords (skrócona wersja)
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'at'
}

def filter_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]
