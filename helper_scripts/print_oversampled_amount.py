import os
import pandas as pd

# Set the directory path for which the numbers of oversampled values should be printed
folder_path = '../unified_datasets'

# --- Process each CSV file ---
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # Filter for oversampled rows
        oversampled = df[df['source'] == 'oversampled']

        # Count by sentiment class
        class_counts = oversampled['sentiment'].value_counts().sort_index()

        print(f"\nFile: {file_name}")
        for sentiment_class in [0, 1, 2]:
            count = class_counts.get(sentiment_class, 0)
            print(f"  Class {sentiment_class}: {count} oversampled")
