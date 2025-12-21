import re
import pandas as pd
import emoji
import string
import contractions
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

missing_strings = ['nan', 'n/a', 'na', '', 'none', 'null']

# Reduce legthening of characters
# E.g. "soooo" -> "soo"
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

# Replace emojis with empty string 
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

# POS tag for lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):    # Adjective
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):  # Verb
        return wordnet.VERB
    elif treebank_tag.startswith('N'):  # Noun
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):  # Adverb
        return wordnet.ADV
    else:                               # Noun
        return wordnet.NOUN


# Text preprocessing
def text_preprocess(text):
    # Step 1: Normalize text
    text_lower = '' if pd.isna(text) else str(text).lower()                                         # Text lowercase
    text_no_mentions = re.sub("@[A-Za-z0-9_]+", "", text_lower)                                     # Remove mentions
    text_no_hashtags = re.sub("#[A-Za-z0-9_]+", "", text_no_mentions)                               # Remove hashtags
    text_no_urls = re.sub(r"http\S+|www\.\S+", "", text_no_hashtags)                                # Remove URLs
    text_expanded = contractions.fix(text_no_urls)                                                  # Expand contractions [E.g. "don't" -> "do not"]
    text_no_numbers = re.sub("[0-9]+", "", text_expanded)                                           # Remove numbers
    text_no_apostrophes = re.sub("'", " ", text_no_numbers)                                         # Replace apostrophes with spaces
    text_no_emojis = remove_emojis(text_no_apostrophes)                                             # Remove emojis
    text_trim = text_no_emojis.translate(str.maketrans('', '', string.punctuation)).strip()         # Remove punctuation & trim
    if re.search(r'[^a-zA-Z\s]', text_trim):                                                        # Remove row of non-alphabetic text
        text_clean = ""
    else:
        text_clean = text_trim

    if len(text_clean) == 0:
        return ""

    # Step 2: Tokenize and process words
    tokens = word_tokenize(text_clean)                                                              # Tokenize Text 
    tokens_reduced = [reduce_lengthening(token) for token in tokens]                                # Reduce repeated characters
    tokens_lemmatized = [lemm.lemmatize(token) for token in tokens_reduced]                         # Lemmatize words

    # Step 3: POS tagging and refined lemmatization
    pos_tags = pos_tag(tokens_lemmatized)                                                           # POS tagging
    tokens_final = [lemm.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]           # Refined lemmatization

    # Step 4: Remove stopwords and short words
    tokens_filtered = [word for word in tokens_final if word not in stop_words and len(word) > 2]   # Remove stopwords [E.g. "the", "is"] & short tokens (length <= 2)

    return " ".join(tokens_filtered)

# Noise detection
def noise_detection(text):

    text = '' if pd.isna(text) else str(text).strip().lower()           # Normalize text

    # Pattern for noise detection
    time_pattern = re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*([ap]\.?m\.?)?\s*$', re.IGNORECASE)                                        # Time formats: 12:30, 1:45 PM, 23:59:59
    date_pattern = re.compile(r'^(\d{1,2}[/\-]\d{1,2}([/\-]\d{2,4})?|\d{1,2}-\w{3,9}|\d{1,2}\s\w{3,9}\s\d{2,4})$', re.IGNORECASE)       # Date formats: 12/31/2020, 31-December-2020, 31 December 2020
    percentage_pattern = re.compile(r'^\s*\d+(\.\d+)?\s*%\s*$', re.IGNORECASE)                                                          # Percentage formats: 45%, 99.9%
    numeric_like_pattern = re.compile(r'^\s*[+-]?\d+([.,]\d+)?\s*$', re.IGNORECASE)                                                     # Numeric-like strings: 123, -45.67, +89, 1,000
    letters_only_pattern = re.sub(r'[^a-zA-Z]', '', text)                                                                               # Letters only: abc123!!! -> abc             
    stripped_pattern = re.sub(r'[^\w]+', '', text)                                                                                      # Stripped text: !!!!!!!, ????
    non_alphanumeric_pattern = re.sub(r'[^a-zA-Z0-9\s]', '', text)                                                                      # Non-alphanumeric characters removed

    # Noise checks, count as noise if True
    if len(text) <= 2:                                                  # Text length <= 2               
        return True
    if numeric_like_pattern.match(text):                                # Numeric-like strings
        return True
    if percentage_pattern.match(text):                                  # Percentage formats
        return True
    if time_pattern.match(text):                                        # Time formats
        return True
    if date_pattern.match(text):                                        # Date formats
        return True
    if len(letters_only_pattern) <= 2:                                  # Letters only length <= 2 
        return True
    if text in missing_strings:                                         # Missing strings
        return True
    if len(stripped_pattern) == 0:                                      # Stripped text length == 0
        return True
    if len(non_alphanumeric_pattern) == 0:                              # Non-alphanumeric characters removed length == 0
        return True
    return False