import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# CONFIGURATION
BASE_PATH = '../intermediate_datasets/handle_class_imbalances'
SAVE_DIR = '../intermediate_datasets/unified_datasets'

DATASET_PATHS = {
    'fast': {
        os.path.join(BASE_PATH, 'airline-tweets-sentiments.csv'): 'airline-tweets-sentiments',
        os.path.join(BASE_PATH, 'Twitter_Tweets_Sentiment_Dataset.csv'): 'Twitter_Tweets_Sentiment',
        os.path.join(BASE_PATH, 'youtube_comments_labeled.csv'): 'youtube_comments',
    },
    'medium': {
        os.path.join(BASE_PATH, 'rotten-tomatoes_audience.csv'): 'rotten-tomatoes_audience',
        os.path.join(BASE_PATH, 'McDonald_s_Reviews.csv'): 'McDonald_s_Reviews',
    },
    'slow': {
        os.path.join(BASE_PATH, 'metacritic_critic_reviews.csv'): 'metacritic_critic',
        os.path.join(BASE_PATH, 'rotten-tomatoes_critics.csv'): 'rotten-tomatoes_critics',
    }
}
ENTRIES_PER_SOURCE = {
    'fast': 18000,
    'medium': 18000,
    'slow': 18000
}
TARGET_CLASSES = [0, 1, 2]
SOURCE_COL = 'source'  # original or oversampled
LABEL_COL = 'sentiment'
TEXT_COL = 'text'


os.makedirs(SAVE_DIR, exist_ok=True)


def load_datasets(file_dict):
    # Read in a dataset
    datasets = {}
    for file, name in file_dict.items():
        df = pd.read_csv(file)
        df['dataset'] = name
        datasets[name] = df
    return datasets


def sample_balanced_per_class(datasets, target_total, source_label):
    # The target numbers of entries that should be collected
    per_class_target = target_total // len(TARGET_CLASSES)
    per_class_per_dataset = per_class_target // len(datasets)

    unified = []

    for cls in TARGET_CLASSES:
        class_entries = []
        remaining = per_class_target

        # Try to collect from each dataset proportionally
        for name, df in datasets.items():
            df_cls = df[df[LABEL_COL] == cls]
            df_cls_original = df_cls[df_cls[SOURCE_COL] == "original"]

            sample_size = min(per_class_per_dataset, len(df_cls_original))
            sampled = df_cls_original.sample(n=sample_size, random_state=42)
            class_entries.append(sampled)
            remaining -= sample_size

        # Fill the gap if remaining samples are needed
        if remaining > 0:
            all_remaining = pd.concat([
                df[df[LABEL_COL] == cls] for df in datasets.values()
            ])
            all_remaining = all_remaining[~all_remaining.index.isin(
                pd.concat(class_entries).index)]
            # Prefer original
            original_pool = all_remaining[all_remaining[SOURCE_COL] == "original"]
            oversampled_pool = all_remaining[all_remaining[SOURCE_COL] == "oversampled"]

            fill_pool = pd.concat([original_pool, oversampled_pool])
            fill_samples = fill_pool.sample(n=min(remaining, len(fill_pool)), random_state=42)
            class_entries.append(fill_samples)

        unified.append(pd.concat(class_entries))

    result_df = pd.concat(unified).sample(frac=1, random_state=42).reset_index(drop=True)
    result_df['datasource'] = source_label
    return result_df


def split_train_test(df):
    train, test = [], []

    for cls in TARGET_CLASSES:
        df_cls = df[df[LABEL_COL] == cls]
        train_cls, test_cls = train_test_split(
            df_cls, test_size=0.2, random_state=42, stratify=None)
        train.append(train_cls)
        test.append(test_cls)

    train_df = pd.concat(train).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test).sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df, test_df


# MAIN PROCESS
all_unified_dfs = []

for source_type, files in DATASET_PATHS.items():
    print(f"Processing source: {source_type}")
    datasets = load_datasets(files)
    unified_df = sample_balanced_per_class(
        datasets, ENTRIES_PER_SOURCE[source_type], source_type)
    all_unified_dfs.append(unified_df)

    train_df, test_df = split_train_test(unified_df)
    train_df.to_csv(f"{SAVE_DIR}/{source_type}_train.csv", index=False)
    test_df.to_csv(f"{SAVE_DIR}/{source_type}_test.csv", index=False)

# Combine all for full unified dataset
final_df = pd.concat(all_unified_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
train_df, test_df = split_train_test(final_df)
train_df.to_csv(f"{SAVE_DIR}/all_sources_train.csv", index=False)
test_df.to_csv(f"{SAVE_DIR}/all_sources_test.csv", index=False)

print("Unified datasets created and split successfully.")
