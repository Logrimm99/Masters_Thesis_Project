import pandas as pd
import time
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

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

output_file = 'svm_output.csv'
output_path = os.path.join(output_dir, output_file)


# Optional: Set best parameters to skip grid search
# best_params = None  # Set to None to run GridSearchCV

# The best parameter combination for the 'all_sources' dataset
best_params = {'svm__C': 1, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 3)}

# The best parameter combination for the fast dataset
#

# The best parameter combination for the medium-fast dataset
#

# The best parameter combination for the slow dataset
#

# Load datasets
df_train = pd.read_csv(os.path.join(input_dir, train_file))
df_test = pd.read_csv(os.path.join(input_dir, test_file))

# Prepare data
df_train['text'] = df_train['text'].fillna('').astype(str)
df_test['text'] = df_test['text'].fillna('').astype(str)

X_train = df_train['text']
y_train = df_train['sentiment']
X_test = df_test['text']
y_test = df_test['sentiment']

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', LinearSVC())
])

start_train = time.time()

if best_params is None:
    # Define parameter grid
    param_grid = {
        'svm__C': [0.01, 0.1, 1, 10],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 3, 5]
    }
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid
    print("Best Parameters from Grid Search:", grid.best_params_)
else:
    # Use provided best parameters
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    model = pipeline
    print("Using manually set best parameters:", best_params)

train_time = time.time() - start_train

# Prediction
start_test = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - start_test

# Save predictions
df_output = df_test.copy()
df_output['predicted_class'] = y_pred
df_output.to_csv(output_path, index=False)

# Print summary
print("Training time: {:.2f} seconds".format(train_time))
print("Testing time: {:.2f} seconds".format(test_time))
print("Classification Report:\n", classification_report(y_test, y_pred))