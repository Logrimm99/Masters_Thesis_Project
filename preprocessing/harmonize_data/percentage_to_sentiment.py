import pandas as pd
import os

input_dir = "../../intermediate_datasets/datasets_percentages"
output_dir = "../../intermediate_datasets/labeled_data"

def assign_sentiment(value, low, high):
    if value <= low:
        return 0
    elif value <= high:
        return 1
    else:
        return 2

def process_file(filepath, output_dir, low_thresh, high_thresh):
    df = pd.read_csv(filepath)

    if "ratings_percentage" not in df.columns:
        print(f"Skipped {filepath}: 'ratings_percentage' column missing.")
        return

    df["sentiment"] = df["ratings_percentage"].apply(lambda x: assign_sentiment(x, low_thresh, high_thresh))

    output_path = os.path.join(output_dir, os.path.basename(filepath))
    # df.to_csv(output_path, index=False)
    df.to_csv(output_path, index=False, encoding="latin1")
    print(f"Saved labeled data to: {output_path}")

def main(input_dir="processed", output_dir="labeled_data"):
    os.makedirs(output_dir, exist_ok=True)

    # Define manual breakpoints for each file
    file_config = [
        {"filename": "McDonald_s_Reviews.csv", "low": 0.26, "high": 0.75},
        {"filename": "rotten-tomatoes_audience.csv", "low": 0.45, "high": 0.78},
        {"filename": "metacritic_critic_reviews.csv", "low": 0.71, "high": 0.81},
        {"filename": "rotten-tomatoes_critics.csv", "low": 0.51, "high": 0.74},
    ]

    for config in file_config:
        filepath = os.path.join(input_dir, config["filename"])
        if os.path.exists(filepath):
            process_file(filepath, output_dir, config["low"], config["high"])
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main(input_dir=input_dir, output_dir=output_dir)
