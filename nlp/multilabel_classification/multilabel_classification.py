from sklearn.preprocessing import MultiLabelBinarizer

from nlp.multilabel_classification.utils import *


def main(method='TFIDF'):
    X_test, X_train, X_val, Y_train, Y_val = load_dataset()
    X_test, X_train, X_val = text_normalization(X_test, X_train, X_val)
    tags_counts, words_counts = count_tags_and_words(X_train, Y_train)

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    Y_train = mlb.fit_transform(Y_train)
    Y_val = mlb.fit_transform(Y_val)

    if method=='BOW':
        x_train, x_val, x_test = transform_to_feature_with_BOW(words_counts, X_train, X_val, X_test)
    elif method=='TFIDF':
        x_train, x_val, x_test, tfidf_vocab = transform_to_feature_with_tfidn(X_train, X_val, X_test)
    else:
        print('Method not supported')
        return

    classifier = train_classifier(x_train, Y_train)

    y_val = classifier.predict(x_val)
    print_evaluation_scores(y_val, Y_val)

    test_prediction = classifier.predict(x_test)
    print(test_prediction)


if __name__ == "__main__":
    main()
