import re
from ast import literal_eval

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = {'are', 'these', 'haven', 'on', 'into', "shan't", 'themselves', 'just', 'each', 'after', 'y', "that'll", "wouldn't", 'not', 'against', 'where', "aren't", 'out', 'o', 'a', 'so', 'in', "isn't", 'once', 'to', 'below', 'will', 'mustn', 'your', 'there', 'own', 'those', 'shan', "mightn't", "you're", "you've", 'had', 'nor', 'most', 'ours', "it's", 'don', 'couldn', 'the', 'very', 'having', 'but', 'all', 'no', 'any', 'hasn', 'myself', 'for', 'because', 'this', 'before', 'who', 'here', "shouldn't", 'itself', 've', 'our', 'then', 'being', "hadn't", 'didn', 'can', 'wasn', 'd', 'some', 's', 'doing', 'too', "you'll", 'from', 'himself', 'you', 'him', 'with', "needn't", 'above', 'won', 'were', 'isn', 'other', 'needn', 'than', 'its', 'few', "you'd", 'during', 'll', 're', 'an', "don't", 'ain', 'my', "haven't", 'yourselves', 'further', 'he', "didn't", 'doesn', 'same', 'whom', 'hers', 'shouldn', 'by', 'yours', 'does', 'wouldn', 'or', 'should', 'yourself', 'her', 'again', 'over', 'such', 'it', "should've", 'ma', 'is', 'up', "mustn't", 'which', 'now', "wasn't", 'weren', 'i', 't', 'she', 'that', 'how', 'under', 'm', 'if', 'be', 'am', "couldn't", 'them', 'they', 'their', 'at', 'aren', 'mightn', 'of', 'theirs', 'hadn', 'both', 'has', 'until', 'off', 'what', 'was', 'did', 'between', 'ourselves', 'through', 'his', 'when', 'down', "weren't", 'as', 'while', "won't", "she's", 'more', 'why', 'been', 'and', 'me', 'only', 'do', "doesn't", "hasn't", 'have', 'about', 'we', 'herself'}

def read_data(file_name):
    data = pd.read_csv(file_name, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text: str) -> str:
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


def transform_to_feature_with_BOW(words_counts, X_train, X_val, X_test):
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


def transform_to_feature_with_tfidn(X_train, X_val, X_test):
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


def print_evaluation_scores(y_val, predicted):

    print(accuracy_score(y_val, predicted))
    print(f1_score(y_val, predicted, average='macro'))
    print(accuracy_score(y_val, predicted))


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary

        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.
    tag_index = tags_classes.index(tag)
    estimator = classifier.estimators_[tag_index]
    estimator.densify()
    coef = estimator.coef_

    sorted_index = np.argpartition(coef, [5, -5], axis=1)
    top_positive_words = [index_to_words[index] for index in sorted_index[0][-5:]]
    top_negative_words = [index_to_words[index] for index in sorted_index[0][:6]]
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


def count_tags_and_words(X_train, Y_train):
    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}
    for tag_list in Y_train:
        for tag in tag_list:
            tags_counts[tag] = tags_counts.get(tag, 0) + 1
    for sentence in X_train:
        words = sentence.split(' ')
        for word in words:
            words_counts[word] = words_counts.get(word, 0) + 1
    return tags_counts, words_counts


def text_normalization(X_test, X_train, X_val):
    X_train = [text_prepare(text) for text in X_train]
    X_val = [text_prepare(text) for text in X_val]
    X_test = [text_prepare(text) for text in X_test]
    return X_test, X_train, X_val


def load_dataset():
    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')
    # print(train.head())
    X_train, Y_train = train['title'].values, train['tags'].values
    X_val, Y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values
    return X_test, X_train, X_val, Y_train, Y_val