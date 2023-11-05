import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import re
import pandas as pd

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Initialize lemmatizer, stemmer and stopwords list
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def remove_symbols(text: str) -> str:
    """remove punctuation, symbols and etc."""

    fix = text
    fix = re.sub(r"\s+", " ", fix)
    fix = re.sub(r"\d+", " ", fix)
    fix = re.sub(r"([.!?])", r" ", fix)
    fix = re.sub(r"[^a-zA-Z.!?]+", r" ", fix)
    fix = fix.strip()
    fix = fix.lower()

    return fix

def preprocess_df(data: pd.DataFrame,
                  toxicity_threshold=0.99
                  ):
    

    mask = data["trn_tox"] > data["ref_tox"]
    temp = data.loc[mask, "reference"].copy()
    data.loc[mask, "reference"] = data.loc[mask, "translation"]
    data.loc[mask, "translation"] = temp

    
    filtered_data = data[
    ((data["ref_tox"] > toxicity_threshold) & (data["trn_tox"] < 1 - toxicity_threshold))
    | ((data["trn_tox"] > toxicity_threshold) & (data["ref_tox"] < 1 - toxicity_threshold))
    ]
    # Preprocess entries for 'reference' and 'translation' columns
    data_preprocessed = filtered_data.copy()
    data_preprocessed['reference'] = data_preprocessed['reference'].apply(remove_symbols)
    data_preprocessed['translation'] = data_preprocessed['translation'].apply(remove_symbols)

    return data_preprocessed




