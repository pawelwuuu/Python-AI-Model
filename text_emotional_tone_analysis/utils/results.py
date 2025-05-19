import csv

def save_sentiment_analysis_results(results, algorithm_name, filename="sentiment_results.csv"):
    """
    Funkcja zapisuje wyniki analizy sentymentu do pliku CSV.
    results: Lista słowników z wynikami analizy sentymentu.
    algorithm_name: Nazwa algorytmu (np. "KNN", "Neural Network").
    filename: Nazwa pliku CSV, do którego zapiszemy dane.
    """
    fieldnames = ['Algorithm', 'Sentence ID', 'Sentiment', 'Confidence']

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        for result in results:
            writer.writerow({
                'Algorithm': algorithm_name,
                'Sentence ID': result['sentence_id'],
                'Sentiment': result['sentiment'],
                'Confidence': result['confidence']
            })
