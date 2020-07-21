import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pickle
import requests
import cv2
import nltk
import missingno
import emoji
import regex
import re
import time

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB

from PIL import Image
from io import BytesIO
from collections import Counter
from gensim import corpora
from numpy import linalg as LA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import word_tokenize

empty_thumbnail_url = 'https://i.ytimg.com/vi/Jw1Y-zhQURU/default.jpg'
def show_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

response = requests.get(empty_thumbnail_url)
empty_thumbnail = np.float32(Image.open(BytesIO(response.content)))

def thumbnail_contrast(url, empty_thumbnail = empty_thumbnail):
    response = requests.get(url)
    img = np.float32(Image.open(BytesIO(response.content)))

    # returning NaN for thumbnails of wrong shape
    if img.shape != empty_thumbnail.shape:
        return np.NaN

    # returning NaN for empty thumbnails
    if not np.any(img - empty_thumbnail):
        return np.NaN

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = img_grey.std()
    return contrast

def process_tags(text):
    text = text.replace('"', " "); text = text.replace('|', "")

    #check that characters not punctuation
    textnopunc = [char for char in text if char not in string.punctuation]

    #join characters back together
    textnopunc = ''.join(textnopunc)

    #convert all to lower
    textnopunc = textnopunc.lower()

    return [word for word in textnopunc.split()]

def letters_per_word(list_of_tags):
    # function that returns the average number of letters in a video tag
    return sum([len(char) for char in list_of_tags])/len(list_of_tags)

def remove_punctuation(text):
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

def check_upper(raw_title):
    '''Detects presence of all capitalised word(s) in title

    Parameters:
    raw_title (str): Raw title string

    Returns:
    (int): Integer number corresponding to boolean presence of capitalised words in title
    '''

    upper = [word for word in raw_title.split() if word.isupper()]

    if len(upper) > 0:
        return 1
    else:
        return 0


def count_caps(raw_title):
    '''Counts number of capitalised words (stop words inclusive) in title

    Parameters:
    raw_title (str): Raw title string

    Returns:
    word_count (int): Integer number of capitalised words in title
    '''

    word_count = 0
    for word in raw_title.split():
        if word.isupper():
            word_count+=1

    return word_count


def count_title(raw_title):
    '''Counts number of words (stop words inclusive) in title

    Parameters:
    raw_title (str): Raw title string

    Returns:
    word_count (int): Integer number of words in title
    '''

    word_count = [word for word in raw_title.split()]

    return len(word_count)

def count_title(raw_title):
    '''Counts number of words (stop words inclusive) in title

    Parameters:
    raw_title (str): Raw title string

    Returns:
    word_count (int): Integer number of words in title
    '''

    word_count = [word for word in raw_title.split()]

    return len(word_count)


# Sentiment intensity of title
def get_sentiment(raw_title):
    '''Calculates compound sentiment score using NLTK VADER Sentiment Analyzer

    Parameters:
    raw_title (str): Raw title string

    Returns:
    sentiment_score (float): Compound value of the sentiment scores, -1 (negative) to +1 (positive)
    '''

    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(raw_title)

    return sentiment_score['compound']


def count_exclaim(raw_title):
    '''Counts number of exclamation marks in title

    Parameters:
    raw_title (str): Raw title string

    Returns:
    counter (int): Integer number of exclamation marks in title
    '''

    counter = raw_title.count(chr(33))
    return counter

def count_question(raw_title):
    '''Counts number of question marks in title

    Parameters:
    raw_title (str): Raw title string

    Returns:
    counter (int): Integer number of question marks in title
    '''

    counter = raw_title.count(chr(63))
    return counter

def count_emoji(raw_title):
    '''Counts number of emojis in title

    Parameters:
    raw_title (str): Raw title string

    Returns:
    counter (int): Integer number of emojis in title
    '''

    counter = 0
    data = regex.findall(r'\X', raw_title)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            counter+=1

    return counter

def sc_transform(df_col):
    # clean text from the column
    df_col = df_col.apply(lambda x: stopwords(remove_punctuation(x)))
    # get frequencies and filter for uncommon words
    freq_d = pd.Series(' '.join(df_col).split()).value_counts()
    rare_words = freq_d[freq_d <= 4]
    freq_d = freq_d[freq_d > 4]
    # compute BoW vector and its norm to get one value
    df_col = df_col.apply(lambda x: [x for x in x.split() if x not in rare_words])
    dict_d = corpora.Dictionary(df_col)
    df_col = df_col.apply(lambda desc: LA.norm(dict_d.doc2bow(desc)))
    return df_col

def desc_lines(description):
    '''Counts number of lines in description, as separated by \n

    Parameters:
    description (str): Description text as a string

    Returns:
    len(lines) (int): Integer number of lines in description
    '''

    lines = description.split('\\n')

    return len(lines)

def desc_urls_count(description):
    '''Counts number of URLs in description, as denoted by http[s]?://

    Parameters:
    description (str): Description text as a string

    Returns:
    url_count (int): Integer number of URLs in description
    '''

    lines = description.split('\\n')

    url_count = 0
    for i in range(len(lines)):

        if re.findall('http[s]?://', lines[i]):
            url_count+=1

    return url_count

nltk_sw = stopwords.words('english')
nltk.download('punkt')

def desc_len(desc):
    ''' Count number of meaningful words

    Parameters:
    desc (str): Description text as a string

    Returns:
    len(cleaned) (int): Number of words in the description
                        after cleaning
    '''
    # remove links and leave text only
    text_only = re.sub(r"http\S+", "", desc)
    text_only = re.sub(r"www\S+", "", text_only)
    text_only = re.sub('[^A-Za-z]+', ' ', text_only)

    # filter out stopwords and very short words
    tokens = word_tokenize(text_only)
    cleaned = [t for t in tokens if t not in nltk_sw]
    cleaned = [w for w in cleaned if len(w) > 3]
    return len(cleaned)

# count the number of capitalized letters
def desc_upper(desc):
    words = re.sub('[^A-Za-z]+', ' ', desc)
    words = words.split()
    count_caps = 0
    for w in words:
        if w.isupper():
            count_caps += 1
    return count_caps

def stopwords(text):
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in nltk_sw]
    #also consider words less than 4 letters as stopwords
    text = [w for w in text if len(w) > 3]
    #remove urls
    d = " ".join(text)
    d = re.sub(r"http\S+", "", d)
    return d
