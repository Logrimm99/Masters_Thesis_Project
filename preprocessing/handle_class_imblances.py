import os
from glob import glob
import pandas as pd
from sklearn.utils import resample
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


input_folder = '../intermediate_datasets/cleaned_texts'
output_folder = '../intermediate_datasets/handle_class_imbalances'

def handleClassImbalances(df, output_path):
    # --- Step 1: Mark original values ---
    valid_labels = [0, 1, 2]

    df["source"] = "original"

    print("Class distribution before oversampling:")
    print(df["sentiment"].value_counts())

    # --- Step 2: Oversample Without Paraphrasing ---
    majority_size = df["sentiment"].value_counts().max()
    dfs = []

    for label in valid_labels:
        subset = df[df["sentiment"] == label]
        if len(subset) < majority_size:
            samples_needed = majority_size - len(subset)
            oversampled = resample(subset, replace=True, n_samples=samples_needed, random_state=42)
            oversampled["source"] = "oversampled"
            subset = pd.concat([subset, oversampled])
        dfs.append(subset)

    # --- Step 3: Combine, Shuffle, Save ---
    df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced.to_csv(output_path, index=False)

    # --- Step 4: Reporting ---
    print("\nClass distribution after oversampling:")
    print(df_balanced["sentiment"].value_counts())
    print("\nSource distribution:")
    print(df_balanced["source"].value_counts())



# Process all CSV files in the folder
os.makedirs(output_folder, exist_ok=True)

# Recursively find all CSV files
csv_files = glob(os.path.join(input_folder, '**', '*.csv'), recursive=True)

for filepath in csv_files:
    try:
        df = pd.read_csv(filepath)
        if 'text' not in df.columns:
            print(f"Skipping {filepath}: 'text' column not found.")
            continue

        print(f"Processing: {os.path.relpath(filepath, input_folder)}")

        # Build mirrored output path
        relative_path = os.path.relpath(filepath, input_folder)
        output_path = os.path.join(output_folder, relative_path)

        # Ensure subdirectory in output path exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        handleClassImbalances(df, output_path)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

