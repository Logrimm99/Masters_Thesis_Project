import pandas as pd
import time
import nltk
import itertools
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn, wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
import os

# --- NLTK downloads ---
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('wordnet')
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

output_file = 'vader_sentiwordnet_output.csv'
output_path = os.path.join(output_dir, output_file)


# Optional: Manually set thresholds to skip optimization
# Format: (positive_threshold, negative_threshold)
# Example: (0.2, -0.1)
# best_thresh = None  # Set to a tuple to use fixed thresholds
best_thresh = (0.05, -0.05)


lemmatizer = WordNetLemmatizer()
vader_analyzer = SentimentIntensityAnalyzer()

# --- POS tag to WordNet POS mapping ---
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None

# --- SentiWordNet scoring ---
def get_swn_score(text):
    tokens = word_tokenize(str(text))
    tagged = pos_tag(tokens)
    score_sum = 0
    count = 0
    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        if not wn_tag:
            continue
        lemma = lemmatizer.lemmatize(word, wn_tag)
        synsets = list(swn.senti_synsets(lemma, wn_tag))
        if synsets:
            s = synsets[0]
            score = s.pos_score() - s.neg_score()
            score_sum += score
            count += 1
    return score_sum / count if count > 0 else 0

# --- VADER scoring ---
def get_vader_score(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

# --- Score to class using thresholds ---
def score_to_class(score, pos_thresh, neg_thresh):
    if score >= pos_thresh:
        return 2
    elif score <= neg_thresh:
        return 0
    return 1

# --- Load training data for optimization ---
train_df = pd.read_csv(os.path.join(input_dir, train_file))
eval_df = pd.read_csv(os.path.join(input_dir, test_file))

print("Computing training scores...")
start_time = time.time()
train_df['combined_score'] = train_df.apply(
    lambda row: get_vader_score(row['text']) if row['datasource'] == 'fast' else get_swn_score(row['text']),
    axis=1
)

# --- Threshold optimization ---
if best_thresh is None:
    print("Searching for best thresholds...")
    threshold_candidates = [x / 100 for x in range(5, 61, 5)]
    best_acc = 0
    best_thresh = (0.05, -0.05)

    for pos_thresh, neg_thresh in itertools.product(threshold_candidates, [-x for x in threshold_candidates]):
        if pos_thresh <= abs(neg_thresh):
            preds = train_df['combined_score'].apply(lambda x: score_to_class(x, pos_thresh, neg_thresh))
            acc = accuracy_score(train_df['sentiment'], preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = (pos_thresh, neg_thresh)

    print(f"Best thresholds found: pos ≥ {best_thresh[0]}, neg ≤ {best_thresh[1]}")
    print(f"Training accuracy: {best_acc:.4f}")
else:
    print(f"Using manually set thresholds: pos ≥ {best_thresh[0]}, neg ≤ {best_thresh[1]}")

# --- Apply to evaluation set ---

print("Computing evaluation scores...")
eval_df['combined_score'] = eval_df.apply(
    lambda row: get_vader_score(row['text']) if row['datasource'] == 'fast' else get_swn_score(row['text']),
    axis=1
)
eval_df['predicted_class'] = eval_df['combined_score'].apply(lambda x: score_to_class(x, *best_thresh))
inference_time = time.time() - start_time

# --- Save output ---
# eval_df.to_csv('../model_predictions/allSources/Vader_SentiWordNets_eval_output.csv', index=False)
eval_df.to_csv(output_path, index=False)

# --- Runtime log ---
print(f"Inference runtime (eval set): {inference_time:.2f} seconds")