import pandas as pd
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import nltk
import emoji
import os
from glob import glob
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

input_folder = '../../intermediate_datasets/subsets'
output_folder = '../../intermediate_datasets/cleaned_texts'

# Make langdetect deterministic
DetectorFactory.seed = 0

# Keep negations
_stop = set(stopwords.words('english'))
stop_words = _stop - {"no", "not", "nor"}

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def remove_noise(df):
    # Clean & Filter Data
    valid_labels = [0, 1, 2]
    df = df[df["sentiment"].isin(valid_labels)]
    df = df[df["text"].notnull() & (df["text"].str.strip() != "")]
    df = df.drop_duplicates()

    def is_english(text):
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    df = df[df["text"].apply(is_english)]
    return df

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    try:
        # Transform emojis to words
        t = emoji.demojize(str(text), language="en")
        t = t.replace(":", " ").replace("_", " ")

        # Transform to lowercase + remove URLs + punctuation => space (keep word boundaries)
        t = t.lower()
        t = re.sub(r'http\S+|www\S+', ' ', t)
        t = re.sub(r"[^\w\s]", " ", t)
        # Tokenize
        tokens = nltk.word_tokenize(t)
        cleaned = []
        for w in tokens:
            # keep alphabetic tokens only
            if w.isalpha() and w not in stop_words:
                # OPTIONAL: comment out next two lines to activate spell-correction
                # corrected = spell.correction(w) or w
                # w = corrected
                # Lemmatize the tokens
                cleaned.append(
                    lemmatizer.lemmatize(w))

        result = ' '.join(cleaned)
        return result.strip()
    except Exception as e:
        print(f"Error cleaning text: {e} → {text}")
        return ""


# Process all CSV files in the folder
os.makedirs(output_folder, exist_ok=True)

for filepath in glob(os.path.join(input_folder, '*.csv')):
    try:
        df = pd.read_csv(filepath)
        if 'text' not in df.columns:
            print(f"Skipping {filepath}: 'text' column not found.")
            continue

        print(f"Processing: {os.path.basename(filepath)}")
        tqdm.pandas(desc=f"Cleaning {os.path.basename(filepath)}")

        df = remove_noise(df)
        df['text'] = df['text'].progress_apply(clean_text)

        output_path = os.path.join(output_folder, os.path.basename(filepath))
        df.to_csv(output_path, index=False)
        print(f"Cleaned text saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
