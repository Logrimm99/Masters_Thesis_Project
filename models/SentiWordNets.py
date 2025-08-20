import pandas as pd
import time
import nltk
from nltk.corpus import sentiwordnet as swn, wordnet
from nltk import pos_tag, word_tokenize
from sklearn.metrics import accuracy_score
import itertools
import os


# --- NLTK downloads ---
nltk.download('punkt', force=True)
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('sentiwordnet')




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

output_file = 'sentiwordnet_output.csv'
output_path = os.path.join(output_dir, output_file)


# Optional: Manually set thresholds to skip optimization
# Format: (positive_threshold, negative_threshold)
# Example: (0.2, -0.1)
# best_thresh = None  # Set to None to enable grid search for thresholds
best_thresh = (0.05, -0.05)


# --- POS mapping ---
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# --- Sentiment score using SentiWordNet ---
def get_sentiment_score(text):
    tokens = word_tokenize(str(text))
    tagged_tokens = pos_tag(tokens)
    total_score = 0
    count = 0

    for word, tag in tagged_tokens:
        wn_tag = get_wordnet_pos(tag)
        if not wn_tag:
            continue
        synsets = list(swn.senti_synsets(word, wn_tag))
        if synsets:
            synset = synsets[0]
            score = synset.pos_score() - synset.neg_score()
            total_score += score
            count += 1

    return total_score / count if count else 0

# --- Class assignment based on thresholds ---
def score_to_class(score, pos_thresh, neg_thresh):
    if score >= pos_thresh:
        return 2
    elif score <= neg_thresh:
        return 0
    else:
        return 1

# --- Load datasets ---
train_df = pd.read_csv(os.path.join(input_dir, train_file))
eval_df = pd.read_csv(os.path.join(input_dir, test_file))

# --- Apply sentiment scores ---
print("Calculating sentiment scores...")
start_time = time.time()
train_df['swn_score'] = train_df['text'].apply(get_sentiment_score)
eval_df['swn_score'] = eval_df['text'].apply(get_sentiment_score)

# --- Threshold optimization ---
if best_thresh is None:
    print("Searching for best thresholds...")
    threshold_candidates = [x / 100 for x in range(5, 61, 5)]
    best_acc = 0
    best_thresh = (0.05, -0.05)

    for pos_thresh, neg_thresh in itertools.product(threshold_candidates, [-x for x in threshold_candidates]):
        if pos_thresh <= abs(neg_thresh):
            preds = train_df['swn_score'].apply(lambda x: score_to_class(x, pos_thresh, neg_thresh))
            acc = accuracy_score(train_df['sentiment'], preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = (pos_thresh, neg_thresh)

    print(f"Best thresholds found: pos ≥ {best_thresh[0]}, neg ≤ {best_thresh[1]}")
    print(f"Training accuracy: {best_acc:.4f}")
else:
    print(f"Using manually set thresholds: pos ≥ {best_thresh[0]}, neg ≤ {best_thresh[1]}")

# --- Apply best thresholds to evaluation data ---
eval_df['predicted_class'] = eval_df['swn_score'].apply(lambda x: score_to_class(x, *best_thresh))
inference_time = time.time() - start_time

# --- Save results ---
# eval_df.to_csv('../model_predictions/allSources/sentiwordnet_eval_output.csv', index=False)
eval_df.to_csv(output_path, index=False)

# --- Timing report ---
print(f"Inference runtime (eval set): {inference_time:.2f} seconds")