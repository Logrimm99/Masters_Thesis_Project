import pandas as pd
import os

from scipy.optimize import direct

# Set the directory path for which the class sizes should be printed
# directory = "../unified_datasets"
# directory = "../original_datasets/fast"
directory = "../intermediate_datasets/labeled_data"

def count_sentiments(file_path, sentiment_column="Sentiment"):
    """
    Reads a CSV file and prints the count and percentage of each sentiment class.
    """
    df = pd.read_csv(file_path, encoding='latin1')

    if sentiment_column not in df.columns:
        sentiment_column = "sentiment"
        if sentiment_column not in df.columns:
            print(f"Column '{sentiment_column}' not found in {file_path}. Skipping.")
            return

    total = len(df)
    sentiment_counts = df[sentiment_column].value_counts().sort_index()
    sentiment_percentages = (sentiment_counts / total * 100).round(2)

    print(f"Sentiment distribution for {file_path}:")
    print(f"Total Length: {total}\n")

    for sentiment in sorted(sentiment_counts.index):
        count = sentiment_counts[sentiment]
        percent = sentiment_percentages[sentiment]
        print(f"Sentiment {sentiment}: {count} ({percent}%)")

    print("-" * 40)

def process_directory(directory):
    """
    Recursively reads all CSV files in the given directory and processes sentiment counts.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            count_sentiments(file_path)


if not os.path.isdir(directory):
    print("Invalid directory path.")
else:
    process_directory(directory)