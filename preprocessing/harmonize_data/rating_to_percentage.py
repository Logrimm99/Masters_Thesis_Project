import pandas as pd
import os
import re
from glob import glob


input_dir = "../../original_datasets"
output_dir = "../../intermediate_datasets/datasets_percentages"

def parse_rating(value, rating_type):
    """Converts various rating formats to a percentage between 0 and 1."""
    if pd.isnull(value):
        return None

    val = str(value).strip().lower()

    try:
        if rating_type == "percent":
            number = float(val)
            return number / 100 if number <= 100 else None

        elif rating_type == "stars_float":
            number = float(val)
            if 0.5 <= number <= 5:
                return (number - 0.5) / (5 - 0.5)  # Normalize from 0.5–5 to 0–1

        elif rating_type == "stars_text":
            match_star = re.match(r"^(\d+(\.\d+)?)\s*(star|stars)?$", val)
            if match_star:
                star_value = float(match_star.group(1))
                if 1 <= star_value <= 5:
                    return (star_value - 1) / (5 - 1)  # Normalize from 1–5 to 0–1

        elif rating_type == "fraction":
            match = re.match(r"^(\d+(\.\d+)?)\s*/\s*(\d+(\.\d+)?)$", val)
            if match:
                num = float(match.group(1))
                denom = float(match.group(3))
                return num / denom if denom != 0 else None

    except:
        return None

    return None  # Unrecognized format

def process_file(filepath, output_dir, rating_col, rating_type):
    df = pd.read_csv(filepath, encoding='latin1')

    if rating_col not in df.columns:
        print(f"Skipped {filepath}: column '{rating_col}' not found.")
        return

    df["ratings_percentage"] = df[rating_col].apply(lambda x: parse_rating(x, rating_type))

    # Drop rows where conversion failed
    df = df[df["ratings_percentage"].notnull()]

    output_path = os.path.join(output_dir, os.path.basename(filepath))
    df.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")


def main(input_dir="datasets", output_dir="processed"):
    os.makedirs(output_dir, exist_ok=True)

    # Manually define how each file should be handled
    file_config = [
        {"filename": "medium-fast/McDonald_s_Reviews.csv", "column": "rating", "type": "stars_text"},
        {"filename": "medium-fast/rotten-tomatoes_audience.csv", "column": "rating", "type": "stars_float"},
        {"filename": "slow/metacritic_critic_reviews.csv", "column": "score", "type": "percent"},
        {"filename": "slow/rotten-tomatoes_critics.csv", "column": "rating", "type": "fraction"},
    ]

    for config in file_config:
        filepath = os.path.join(input_dir, config["filename"])
        if os.path.exists(filepath):
            process_file(filepath, output_dir, config["column"], config["type"])
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main(input_dir=input_dir, output_dir=output_dir)
