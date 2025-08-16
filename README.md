# Emotional Tone Analysis of Text

The project is designed for classifying the emotional tone of text messages using various machine learning models: Naive Bayes, KNN, SVM, and a neural network.

## 📁 Directory Structure

```
project2/
├── text_emotional_tone_analysis/
│   ├── data/                  # Training and test data
│   │   ├── text_messages.py
│   │   └── train_data.py
│   ├── models/                # ML model implementations
│   │   ├── knn_model.py
│   │   ├── naive_bayes_model.py
│   │   ├── sentiment_model.py
│   │   └── svm_model.py
│   ├── utils/                 # Tokenization, metrics, visualizations
│   │   ├── results.py
│   │   ├── tokenizer.py
│   │   └── visualizations.py
│   ├── main_bayes.py          # Run Naive Bayes classification
│   ├── main_knn.py            # Run KNN classification
│   ├── main_svm.py            # Run SVM classification
│   ├── main_neural.py         # Run neural network
│   ├── generate_accuracy_chart.py
│   └── generate_confidence_chart.py
├── packages.txt               # List of required packages
└── README.md                  # Documentation
```

## 🧠 Models

The project implements and compares different emotional tone classifiers:

- **Naive Bayes** – fast and effective for text
- **KNN (K-Nearest Neighbors)** – similarity-based classification
- **SVM (Support Vector Machine)** – linear classifier with margin
- **Neural Network** – model based on Transformers/Pipeline

## 📦 Requirements

To run the project, install the dependencies:

```bash
pip install -r packages.txt
```

## 🚀 Running

Each model is executed via a separate script:

```bash
python text_emotional_tone_analysis/main_bayes.py
python text_emotional_tone_analysis/main_knn.py
python text_emotional_tone_analysis/main_svm.py
python text_emotional_tone_analysis/main_neural.py
```

## 📊 Visualizations

To generate accuracy and confidence charts:

```bash
python text_emotional_tone_analysis/generate_accuracy_chart.py
python text_emotional_tone_analysis/generate_confidence_chart.py
```

Each of the `main_*.py` files (e.g., `main_bayes.py`, `main_knn.py`) trains the model and generates classification results, which are saved into a CSV file. This file contains data such as accuracy, predicted labels, classification probabilities, and other metrics.

The scripts then create plots that present model performance comparisons and classification confidence distributions.

## 📝 Data

The files `data/text_messages.py` and `data/train_data.py` contain sample data used for training and testing the models.

## 🛠 Useful Modules

* `tokenizer.py` – text data preprocessing  
* `results.py` – metrics and evaluation  
* `visualizations.py` – generating result charts  

## 👤 Authors

Project authors: **Igor Rozanowski**, **Pawel Wojcik**
