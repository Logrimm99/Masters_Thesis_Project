import pandas as pd
import numpy as np
import os
from glob import glob

data_dir = "../../intermediate_datasets/datasets_percentages"
score_col = "ratings_percentage"

def classify_scores(scores, low_thresh, high_thresh):
    return scores.apply(lambda x: 0 if x <= low_thresh else (1 if x <= high_thresh else 2))

def evaluate_all_breakpoints(scores, step=0.01):
    results = []
    for low in np.arange(0.1, 0.98, step):
        for high in np.arange(low + step, 0.99, step):
            classes = classify_scores(scores, low, high)
            proportions = classes.value_counts(normalize=True).sort_index()
            props = [proportions.get(i, 0.0) for i in range(3)]

            # Compute total absolute deviation from perfect balance (33.33% per class)
            deviation = sum([abs(p - 1/3) for p in props])
            results.append({
                "low": float(low),
                "high": float(high),
                "class_0": round(props[0], 4),
                "class_1": round(props[1], 4),
                "class_2": round(props[2], 4),
                "imbalance": round(deviation, 4)
            })
    return results

def get_best(results):
    imbalance = [r["imbalance"] for r in results]
    min_idx = np.argmin(imbalance)
    return results[min_idx]

def process_file(filepath, step=0.01):
    df = pd.read_csv(filepath)
    if score_col not in df.columns:
        raise ValueError(f"Missing '{score_col}' column in {filepath}")

    df = df[df[score_col].notnull()]  # Ensure no missing values
    scores = df[score_col].astype(float)
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]

    print(f"\nProcessing {dataset_name} — total rows: {len(scores)}")

    results = evaluate_all_breakpoints(scores, step)
    best = get_best(results)

    # Re-apply the best thresholds to confirm the actual distribution
    labeled = classify_scores(scores, best['low'], best['high'])
    actual_dist = labeled.value_counts(normalize=True).sort_index().to_dict()

    print(f"Best Balanced Threshold for {dataset_name}:")
    print(f"  Breakpoints: low = {best['low']}, high = {best['high']}")
    print(f"  Target Distribution: [Negative: {best['class_0']:.2%}, Neutral: {best['class_1']:.2%}, Positive: {best['class_2']:.2%}]")
    print(f"  Actual Labeled Distribution: [", end="")
    for i in range(3):
        pct = actual_dist.get(i, 0.0)
        print(f"{['Negative','Neutral','Positive'][i]}: {pct:.2%} ", end="")
    print("]")
    print(f"  Total Deviation from Perfect Balance: {best['imbalance']:.4f}")

def main(data_dir="datasets", step=0.01):
    filepaths = glob(os.path.join(data_dir, "*.csv"))
    for filepath in filepaths:
        try:
            process_file(filepath, step)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

# Example usage
if __name__ == "__main__":
    main(data_dir=data_dir, step=0.01)
