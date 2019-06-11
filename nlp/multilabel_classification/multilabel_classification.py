import pandas as pd
from nlp.multilabel_classification.utils import read_data, text_prepare

train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')

# print(train.head())

X_train, Y_train = train['title'].values, train['tags'].values
X_val, Y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

X_train = [text_prepare(text) for text in X_train]
X_val = [text_prepare(text) for text in X_val]
X_test = [text_prepare(text) for text in X_test]


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


DICT_SIZE = 5000
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE]
WORDS_TO_INDEX = {word: index for index, (word, frequency) in enumerate(most_common_words)}
# INDEX_TO_WORDS =  ####### YOUR CODE HERE #######
ALL_WORDS = WORDS_TO_INDEX.keys()

