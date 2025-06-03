# Analiza Emocjonalnego Tonu Tekstu

Projekt służy do klasyfikacji emocjonalnego tonu wiadomości tekstowych przy użyciu różnych modeli uczenia maszynowego: Naive Bayes, KNN, SVM oraz sieci neuronowej.

## 📁 Struktura katalogów

```

projekt2/
├── text\_emotional\_tone\_analysis/
│   ├── data/                  # Dane treningowe i testowe
│   │   ├── text\_messages.py
│   │   └── train\_data.py
│   ├── models/                # Implementacje modeli ML
│   │   ├── knn\_model.py
│   │   ├── naive\_bayes\_model.py
│   │   ├── sentiment\_model.py
│   │   └── svm\_model.py
│   ├── utils/                 # Tokenizacja, metryki, wizualizacje
│   │   ├── results.py
│   │   ├── tokenizer.py
│   │   └── visualizations.py
│   ├── main\_bayes.py          # Uruchomienie klasyfikacji Naive Bayes
│   ├── main\_knn.py            # Uruchomienie klasyfikacji KNN
│   ├── main\_svm.py            # Uruchomienie klasyfikacji SVM
│   ├── main\_neural.py         # Uruchomienie sieci neuronowej
│   ├── generate\_accurancy\_chart.py
│   └── generate\_confidence\_chart.py
├── packages.txt               # Lista wymaganych pakietów
└── README.md                  # Dokumentacja

````

## 🧠 Modele

Projekt implementuje i porównuje różne klasyfikatory emocjonalnego tonu:

- **Naive Bayes** – szybki i skuteczny dla tekstów
- **KNN (K-Nearest Neighbors)** – klasyfikacja oparta na podobieństwie
- **SVM (Support Vector Machine)** – klasyfikator liniowy z marginesem
- **Sieć neuronowa** – model oparty na Transformers/Pipeline

## 📦 Wymagania

Aby uruchomić projekt, zainstaluj zależności:

```bash
pip install -r packages.txt
````

## 🚀 Uruchamianie

Każdy model uruchamiany jest osobnym skryptem:

```bash
python text_emotional_tone_analysis/main_bayes.py
python text_emotional_tone_analysis/main_knn.py
python text_emotional_tone_analysis/main_svm.py
python text_emotional_tone_analysis/main_neural.py
```

## 📊 Wizualizacje

Aby wygenerować wykresy dokładności i pewności klasyfikacji:

```bash
python text_emotional_tone_analysis/generate_accurancy_chart.py
python text_emotional_tone_analysis/generate_confidence_chart.py
```

Każdy z plików main_*.py (np. main_bayes.py, main_knn.py) po uruchomieniu trenuje model i generuje wyniki klasyfikacji, które są zapisywane do pliku CSV. Plik ten zawiera dane takie jak dokładność, przewidywane etykiety, prawdopodobieństwo klasyfikacji i inne metryki.

Następnie za pomocą skryptów tworzone są wykresy, które prezentują porównanie skuteczności modeli oraz rozkład pewności klasyfikacji.

## 📝 Dane

W plikach `data/text_messages.py` i `data/train_data.py` znajdują się przykładowe dane używane do trenowania i testowania modeli.

## 🛠 Przydatne moduły

* `tokenizer.py` – przygotowanie danych tekstowych
* `results.py` – metryki i ewaluacja
* `visualizations.py` – generowanie wykresów z wynikami

## 👤 Autor

Autor projektu: Igor Rozanowski, Pawel Wojcik
