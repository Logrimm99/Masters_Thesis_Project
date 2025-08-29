import pandas as pd
import os

# Input and output file paths
input_file = '../../original_datasets/fast/Twitter_Tweets_Sentiment_Dataset.csv'
output_file = '../../intermediate_datasets/labeled_data/Twitter_Tweets_Sentiment_Dataset.csv'

# Sentiment mapping
sentiment_map = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

# Read the input CSV
try:
    df = pd.read_csv(input_file)

    if 'sentiment' not in df.columns:
        raise ValueError("Missing 'sentiment' column in input file.")

    # Map sentiments to numeric values
    df['sentiment'] = df['sentiment'].str.strip().str.lower().map(sentiment_map)

    # Save the transformed dataset
    df.to_csv(output_file, index=False)
    print(f"Saved transformed file to: {output_file}")

except Exception as e:
    print(f"Error processing file: {e}")
