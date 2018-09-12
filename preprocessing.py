import numpy as np
from sklearn import preprocessing

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

    return preprocessing.scale(features), labels
