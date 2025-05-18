# models/knn_model.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Funkcja do trenowania modelu KNN
def train_knn_model(X, y, n_neighbors=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Zmiana: używamy predict_proba zamiast predict
    y_proba = knn.predict_proba(X_test)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return knn, accuracy, y_proba  # Zwracamy także prawdopodobieństwa