import pandas as pd
import time
import itertools
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import os


# CONFIGURATION
input_dir = '../unified_datasets'

data_type = 'allSources'
# data_type = 'fast'
# data_type = 'mediumFast'
# data_type = 'slow'

match data_type:
    case 'allSources':
        train_file = 'all_sources_train.csv'
        test_file = 'all_sources_test.csv'
        output_dir = '../model_predictions/allSources'
    case 'fast':
        train_file = 'fast_train.csv'
        test_file = 'fast_test.csv'
        output_dir = '../model_predictions/fast'
    case 'mediumFast':
        train_file = 'medium_train.csv'
        test_file = 'medium_test.csv'
        output_dir = '../model_predictions/mediumFast'
    case 'slow':
        train_file = 'slow_train.csv'
        test_file = 'slow_test.csv'
        output_dir = '../model_predictions/slow'

output_file = 'vader_output.csv'
output_path = os.path.join(output_dir, output_file)


# Optional: manually set thresholds to skip optimization
# Format: (positive_threshold, negative_threshold)
# Example: (0.25, -0.15)
# best_thresh = None  # Set to a tuple to use fixed thresholds
best_thresh = (0.05, -0.05)


# --- Start total timing ---
total_start_time = time.time()

# --- Load datasets ---
train_df = pd.read_csv(os.path.join(input_dir, train_file))
eval_df = pd.read_csv(os.path.join(input_dir, test_file))

# --- Initialize VADER ---
analyzer = SentimentIntensityAnalyzer()

# --- Apply VADER compound scores ---
start_time = time.time()
train_df['compound_score'] = train_df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
eval_df['compound_score'] = eval_df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# --- Thresholding function ---
def compound_to_class(score, pos_thresh, neg_thresh):
    if score >= pos_thresh:
        return 2
    elif score <= neg_thresh:
        return 0
    else:
        return 1

# --- Threshold optimization ---
if best_thresh is None:
    print("Searching for best thresholds...")
    threshold_candidates = [x / 100 for x in range(5, 61, 5)]
    best_acc = 0
    best_thresh = (0.05, -0.05)  # Default VADER values

    for pos_thresh, neg_thresh in itertools.product(threshold_candidates, [-x for x in threshold_candidates]):
        if pos_thresh <= abs(neg_thresh):
            preds = train_df['compound_score'].apply(lambda x: compound_to_class(x, pos_thresh, neg_thresh))
            acc = accuracy_score(train_df['sentiment'], preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = (pos_thresh, neg_thresh)

    print(f"Best thresholds found: pos ≥ {best_thresh[0]}, neg ≤ {best_thresh[1]}")
    print(f"Training accuracy: {best_acc:.4f}")
else:
    print(f"Using manually set thresholds: pos ≥ {best_thresh[0]}, neg ≤ {best_thresh[1]}")

# --- Apply best thresholds to evaluation data ---
eval_df['predicted_class'] = eval_df['compound_score'].apply(lambda x: compound_to_class(x, *best_thresh))
inference_time = time.time() - start_time

# --- Save results ---
eval_df.to_csv(output_path, index=False)

# --- Runtime summary ---
total_runtime = time.time() - total_start_time
print(f"Inference runtime (eval set): {inference_time:.2f} seconds")
print(f"Total runtime: {total_runtime:.2f} seconds")