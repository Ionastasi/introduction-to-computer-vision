from sklearn.svm import SVC
from sklearn import cross_validation

def fit_and_classify(train_featues, train_labels, test_features):
    classifier = SVC(kernel='linear')
    classifier.fit(train_featues, train_labels)
    test_labels = classifier.predict(test_features)
    return test_labels
