from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from read_csv import read
from preprocessing import split_features_labels, split_train_test

fname = './UCI_Credit_Card.csv'

def linebreak():
    print('\n')
    print('-' * 100)
    print('\n')

def svm(train_features, train_labels, test_features, test_labels):
    clf = SVC()
    print('Start training svm...')
    tStart = time()
    clf.fit(train_features, train_labels)
    print('Training time: ', round(time()-tStart, 3), 's')
    print('Accuracy: ', accuracy_score(clf.predict(test_features), test_labels))

def lr(train_features, train_labels, test_features, test_labels):
    clf = LogisticRegression(random_state=0)
    print('Start training logistic regression...')
    tStart = time()
    clf.fit(train_features, train_labels)
    print('Training time: ', round(time()-tStart, 3), 's')
    print('Accuracy: ', accuracy_score(clf.predict(test_features), test_labels))

def main():
    data_set = read(fname)
    features, labels = split_features_labels(data_set)
    train_features, train_labels, test_features, test_labels = split_train_test(features, labels, 0.7)
    print('training samples: %d\ntesting samples: %d' % (len(train_features), len(test_features)))

    linebreak()
    svm(train_features, train_labels, test_features, test_labels)
    linebreak()

    linebreak()
    lr(train_features, train_labels, test_features, test_labels)
    linebreak()

if __name__ == '__main__':
    main()
