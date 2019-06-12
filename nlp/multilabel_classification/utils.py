import re
from ast import literal_eval
import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = {'are', 'these', 'haven', 'on', 'into', "shan't", 'themselves', 'just', 'each', 'after', 'y', "that'll", "wouldn't", 'not', 'against', 'where', "aren't", 'out', 'o', 'a', 'so', 'in', "isn't", 'once', 'to', 'below', 'will', 'mustn', 'your', 'there', 'own', 'those', 'shan', "mightn't", "you're", "you've", 'had', 'nor', 'most', 'ours', "it's", 'don', 'couldn', 'the', 'very', 'having', 'but', 'all', 'no', 'any', 'hasn', 'myself', 'for', 'because', 'this', 'before', 'who', 'here', "shouldn't", 'itself', 've', 'our', 'then', 'being', "hadn't", 'didn', 'can', 'wasn', 'd', 'some', 's', 'doing', 'too', "you'll", 'from', 'himself', 'you', 'him', 'with', "needn't", 'above', 'won', 'were', 'isn', 'other', 'needn', 'than', 'its', 'few', "you'd", 'during', 'll', 're', 'an', "don't", 'ain', 'my', "haven't", 'yourselves', 'further', 'he', "didn't", 'doesn', 'same', 'whom', 'hers', 'shouldn', 'by', 'yours', 'does', 'wouldn', 'or', 'should', 'yourself', 'her', 'again', 'over', 'such', 'it', "should've", 'ma', 'is', 'up', "mustn't", 'which', 'now', "wasn't", 'weren', 'i', 't', 'she', 'that', 'how', 'under', 'm', 'if', 'be', 'am', "couldn't", 'them', 'they', 'their', 'at', 'aren', 'mightn', 'of', 'theirs', 'hadn', 'both', 'has', 'until', 'off', 'what', 'was', 'did', 'between', 'ourselves', 'through', 'his', 'when', 'down', "weren't", 'as', 'while', "won't", "she's", 'more', 'why', 'been', 'and', 'me', 'only', 'do', "doesn't", "hasn't", 'have', 'about', 'we', 'herself'}

def read_data(file_name):
    data = pd.read_csv(file_name, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    # lowercase text
    text = text.lower()
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, '', text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    # delete stopwords from text
    text_list = text.split(' ')
    text_list = [word for word in text_list if word not in STOPWORDS and word != '']
    text = str.join(' ', text_list)
    return text


def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    for word in text.split(' '):
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def pure_bag_of_words(words_counts, X_train, X_val, X_test):
    DICT_SIZE = 5000
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {word: index for index, (word, frequency) in enumerate(most_common_words)}
    # INDEX_TO_WORDS =  ####### YOUR CODE HERE #######
    ALL_WORDS = WORDS_TO_INDEX.keys()
    X_train_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
    X_val_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
    X_test_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
    print('X_train shape ', X_train_mybag.shape)
    print('X_val shape ', X_val_mybag.shape)
    print('X_test shape ', X_test_mybag.shape)
    return X_train_mybag, X_val_mybag, X_test_mybag


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer('content', max_df=0.9, min_df=5, token_pattern='(\S+)')

    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    estimator = LogisticRegression()
    classifier = OneVsRestClassifier(estimator)
    classifier.fit(X_train, y_train)
    return classifier


