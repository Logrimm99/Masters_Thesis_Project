import pandas as pd
import os
from glob import glob

input_folder = "../../intermediate_datasets/labeled_data"
output_folder = "../../intermediate_datasets/subsets"


def create_subset(input_file, output_file, label_col='sentiment', max_per_class=10000):
    # Read the dataset
    df = pd.read_csv(input_file)

    # Check if the label column exists
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {input_file}")

    # Sample up to max_per_class per sentiment class
    subset_list = []
    for sentiment_class in sorted(df[label_col].dropna().unique()):
        class_subset = df[df[label_col] == sentiment_class].sample(
            n=min(max_per_class, (df[label_col] == sentiment_class).sum()),
            random_state=42
        )
        subset_list.append(class_subset)

    # Combine and shuffle the subset
    subset_df = pd.concat(subset_list).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    subset_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Subset saved to: {output_file}")
    print(subset_df[label_col].value_counts())

# Process all labeled datasets
if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    for filepath in glob(os.path.join(input_folder, "*.csv")):
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_folder, filename)
        try:
            create_subset(filepath, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")