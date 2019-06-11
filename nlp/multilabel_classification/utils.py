import re
from ast import literal_eval
import numpy as np
import pandas as pd

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
        result_vector[words_to_index[word]] += 1
    return result_vector