from time import time
import numpy as np
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score
from read_csv import read

fname = 'UCI_Credit_Card.csv'

def main():
    data_set = read(fname)
    features, labels = split_features_labels(data_set)
    train_features, train_labels, test_features, test_labels = split_train_test(features, labels, 0.7)
    print(len(train_features), ' ', len(test_features))
    clf = svm.SVC()
    print('Start training...')
    tStart = time()
    clf.fit(train_features, train_labels)
    print('Training time: ', round(time()-tStart, 3), 's')
    print('Accuracy: ', accuracy_score(clf.predict(test_features), test_labels))

def split_train_test(features, labels, test_size):
    total_test_size = int(len(features) * test_size)
    np.random.seed(2)
    indices = np.random.permutation(len(features))
    train_features = features[indices[:-total_test_size]]
    train_labels = labels[indices[:-total_test_size]]
    test_features  = features[indices[-total_test_size:]]
    test_labels  = labels[indices[-total_test_size:]]
    return train_features, train_labels, test_features, test_labels

def split_features_labels(data_set):
    features = data_set['features']
    labels = data_set['labels']

    ids_removed = [x[1:] for x in features]
    features = [x[:11] for x in ids_removed]

    bills = [x[11:17] for x in ids_removed]
    payments = [x[17:] for x in ids_removed]

    div = lambda a,b: [x/y if y > 0 else 1 for x, y in zip(a, b)]
    fraction_paid = list(map(div, payments, bills))

    features = [val+fraction_paid[idx] for idx,val in enumerate(features)]

    features = np.array([np.array(x) for x in features])
    labels = np.array(labels)

    #scaler = preprocessing.MinMaxScaler(feature_range=(-2, 9))

    scaled_limit_balance = preprocessing.scale(features[:,0])
    scaled_age = preprocessing.scale(features[:,4])

    features[:,0] = scaled_limit_balance
    features[:,4] = scaled_age

    return features, labels

if __name__ == '__main__':
    main()
