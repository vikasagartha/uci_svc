from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from read_csv import read
from preprocessing import split_features_labels, split_train_test

fname = './UCI_Credit_Card.csv'

results = {
        'meta': {
            'samples_trained': 0,
            'samples_tested': 0
            },
        'models': []
        }

def svm(train_features, train_labels, test_features, test_labels):
    clf = SVC()
    tStart = time()
    clf.fit(train_features, train_labels)

    results['models'].append({
        'name': 'svm',
        'training_time': round(time()-tStart, 3),
        'accuracy': accuracy_score(clf.predict(test_features), test_labels)
        })

def lr(train_features, train_labels, test_features, test_labels):
    clf = LogisticRegression(random_state=0)
    tStart = time()
    clf.fit(train_features, train_labels)

    results['models'].append({
        'name': 'logistic regression',
        'training_time': round(time()-tStart, 3),
        'accuracy': accuracy_score(clf.predict(test_features), test_labels)
        })

def main():
    data_set = read(fname)
    features, labels = split_features_labels(data_set)
    train_features, train_labels, test_features, test_labels = split_train_test(features, labels, 0.7)

    results['meta']['samples_trained'] = len(train_features)
    results['meta']['samples_tested'] = len(test_features)

    print('Models training.....')

    svm(train_features, train_labels, test_features, test_labels)
    lr(train_features, train_labels, test_features, test_labels)

if __name__ == '__main__':
    main()
    print('training samples: %d\ntesting samples: %d' % (results['meta']['samples_trained'], results['meta']['samples_tested']))
    [print(res) for res in results['models']]
