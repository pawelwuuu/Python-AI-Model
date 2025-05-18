from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.utils.validation import check_is_fitted
from collections import Counter

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='distance', metric='cosine'):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            algorithm='auto'
        )
        self.classes_ = None

    def _can_stratify(self, y):
        class_counts = Counter(y)
        return all(count >= 2 for count in class_counts.values())

    def train(self, X, y, test_size=0.2, random_state=42):
        stratify = y if self._can_stratify(y) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return accuracy, report

    def predict_sentiment(self, X):
        check_is_fitted(self.model)
        
        try:
            proba = self.model.predict_proba(X)[0]
        except AttributeError:
            pred = self.model.predict(X)[0]
            proba = [1.0 if c == pred else 0.0 for c in self.classes_]
        
        idx = np.argmax(proba)
        return self.classes_[idx], proba[idx]

    def get_class_distribution(self, y):
        return Counter(y)