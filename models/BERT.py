import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import os

# Enable GPU growth to optimize GPU usage in Colab
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configuration
MAX_LEN = 512
EPOCHS = 5
LEARNING_RATE = 3e-5
FREEZE_BERT = True

BATCH_SIZE = 32
optimize_hyperparameters = False  # Set to True to enable grid search

# Paths
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

output_file = 'bert_output.csv'
output_path = os.path.join(output_dir, output_file)

# --- Load datasets ---
df_train = pd.read_csv(os.path.join(input_dir, train_file))
df_test = pd.read_csv(os.path.join(input_dir, test_file))

X_train_text_full = df_train['text'].fillna("").tolist()
y_train_full = df_train['sentiment'].tolist()
X_test_text = df_test['text'].fillna("").tolist()
y_test = df_test['sentiment'].tolist()

# --- Tokenization ---
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# --- Model builder function ---
def build_model(lr=2e-5, freeze_bert=False, max_len=256):
    bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", from_pt=True)
    if freeze_bert:
        bert_model.trainable = False

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    def extract_bert_hidden_states(inputs):
        return bert_model(
            input_ids=inputs[0],
            attention_mask=inputs[1]
        ).last_hidden_state

    bert_output = Lambda(extract_bert_hidden_states, output_shape=(max_len, 768))([input_ids, attention_mask])
    x = Bidirectional(LSTM(128, return_sequences=False))(bert_output)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(3, activation="softmax")(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# --- Hyperparameter tuning ---
if optimize_hyperparameters:
    print("Starting hyperparameter search...")

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_text_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    param_grid = {
        "epochs": [3, 5],
        "learning_rate": [2e-5, 3e-5],
        "freeze_bert": [True, False],
        "max_len": [128, 256, 512]
    }

    best_acc = 0
    best_config = None

    for epochs in param_grid["epochs"]:
        for lr in param_grid["learning_rate"]:
            for freeze in param_grid["freeze_bert"]:
                for max_len in param_grid["max_len"]:
                    print(f"Testing config: epochs={epochs}, lr={lr}, freeze_bert={freeze}, max_len={max_len}")
                    tokenizer.model_max_length = max_len

                    train_tokens = tokenizer(X_train_split, padding=True, truncation=True, max_length=max_len,
                                             return_tensors="tf")
                    val_tokens = tokenizer(X_val_split, padding=True, truncation=True, max_length=max_len,
                                           return_tensors="tf")

                    model = build_model(lr, freeze, max_len)
                    model.fit(
                        {"input_ids": train_tokens["input_ids"], "attention_mask": train_tokens["attention_mask"]},
                        np.array(y_train_split),
                        validation_data=(
                            {"input_ids": val_tokens["input_ids"], "attention_mask": val_tokens["attention_mask"]},
                            np.array(y_val_split)
                        ),
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],
                        verbose=0
                    )

                    val_preds = np.argmax(model.predict({
                        "input_ids": val_tokens["input_ids"],
                        "attention_mask": val_tokens["attention_mask"]
                    }), axis=1)

                    acc = accuracy_score(y_val_split, val_preds)
                    print(f"→ Validation Accuracy: {acc:.4f}")

                    if acc > best_acc:
                        best_acc = acc
                        best_config = (epochs, lr, freeze, max_len)
                        print("New best config:", best_config)

    EPOCHS, LEARNING_RATE, FREEZE_BERT, MAX_LEN = best_config
    print(f"Best config: epochs={EPOCHS}, lr={LEARNING_RATE}, freeze_bert={FREEZE_BERT}, max_len={MAX_LEN}")

# --- Final tokenization ---
tokenizer.model_max_length = MAX_LEN
X_train_tokens = tokenizer(X_train_text_full, padding='max_length', truncation=True, max_length=MAX_LEN,
                           return_tensors="tf")
X_test_tokens = tokenizer(X_test_text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="tf")

# --- Final training with best or fixed hyperparameters ---
model = build_model(LEARNING_RATE, FREEZE_BERT, MAX_LEN)
callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]

start_train = time.time()
model.fit(
    {"input_ids": X_train_tokens["input_ids"], "attention_mask": X_train_tokens["attention_mask"]},
    np.array(y_train_full),
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)
end_train = time.time()
print(f"Training time: {end_train - start_train:.2f} seconds")

# --- Prediction ---
start_test = time.time()
y_pred_probs = model.predict({
    "input_ids": X_test_tokens["input_ids"],
    "attention_mask": X_test_tokens["attention_mask"]
})
y_pred = np.argmax(y_pred_probs, axis=1)
end_test = time.time()
print(f"Testing time: {end_test - start_test:.2f} seconds")

# --- Save results ---
df_test = df_test.copy()
df_test['predicted_class'] = y_pred
df_test.to_csv(output_path, index=False)

print(f"Results saved to '{output_path}'")