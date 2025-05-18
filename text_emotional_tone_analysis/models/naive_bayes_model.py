# models/naive_bayes_model.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_naive_bayes_model(X, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, y_proba 