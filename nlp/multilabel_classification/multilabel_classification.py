import pandas as pd
from nlp.multilabel_classification.utils import *


def main():
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
    X_train_mybag, X_val_mybag, X_test_mybag = pure_bag_of_words(words_counts, X_train, X_val, X_test)

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)

    classifier_mybag = train_classifier(X_train_mybag, Y_train)
    y_val_mybag = classifier_mybag.predict(X_val_mybag)

    classifier_tfidf = train_classifier(X_train_tfidf, Y_train)
    y_val_tfidf = classifier_tfidf.predict(X_val_tfidf)


main()




