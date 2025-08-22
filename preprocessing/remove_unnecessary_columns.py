import pandas as pd
import os

# CONFIGURATION
INPUT_DIR = '../intermediate_datasets/unified_datasets'
OUTPUT_DIR = '../unified_datasets'
FILENAMES = [
    'fast_train.csv', 'fast_test.csv',
    'medium_train.csv', 'medium_test.csv',
    'slow_train.csv', 'slow_test.csv',
    'all_sources_train.csv', 'all_sources_test.csv'
]
COLUMNS_TO_KEEP = ['text', 'sentiment', 'source', 'datasource']

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in FILENAMES:
    input_path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(input_path)
    df_trimmed = df[COLUMNS_TO_KEEP]
    output_path = os.path.join(OUTPUT_DIR, filename)
    df_trimmed.to_csv(output_path, index=False)


print("Trimmed datasets saved successfully.")
