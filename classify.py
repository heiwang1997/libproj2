#!/usr/bin/env python3
# -*- coding=utf-8 -*-

"""
Python script implementing task 2 in project 2, Data Mining.
Comparison of different classifiers.
"""

import scipy.io
from numpy import genfromtxt
import numpy as np
import pickle
import time
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict


def get01label(class_mtx, class_id):
    label = np.zeros((len(class_mtx),))
    for i, c in enumerate(class_mtx):
        if class_id in c:
            label[i] = 1
    return label

def get_weighted_score(score_arr):
    all_weights = sum(map(lambda x: x[0], score_arr))
    prec = 0
    recal = 0
    f1_measure = 0
    all_time = 0
    for i in score_arr:
        this_weight = i[0] / all_weights
        prec += this_weight * i[1]
        recal += this_weight * i[2]
        f1_measure += this_weight * i[3]
        all_time += i[4]
    return prec, recal, f1_measure, all_time

if __name__ == '__main__':

    feat_matrix = scipy.io.mmread("feat.mtx")
    feat_matrix = feat_matrix.tocsr()
    all_document_count = feat_matrix.shape[0]
    train_count = int(all_document_count * 0.9)
    test_count = all_document_count - train_count

    with open("Class.pkl", "rb") as f:
        [class_name, class_matrix] = pickle.load(f)

    # Shuffle is applied in preprocessing.
    X_train = feat_matrix[0:train_count, :].toarray()
    X_test = feat_matrix[train_count:, :].toarray()
    class_matrix_train = class_matrix[0:train_count]
    class_matrix_test = class_matrix[train_count:]

    class_name = sorted(list(class_name.items()), key=lambda x: x[1])
    estimator_score = defaultdict(list)

    for [cname, cid] in class_name:
        print("Running Classifier for class %d - %s" % (cid, cname))
        y_train = get01label(class_matrix_train, cid)
        y_test = get01label(class_matrix_test, cid)

        y_true_count = np.count_nonzero(y_test)
        if y_true_count == 0:
            print("========= NO TRUE SAMPLES ==========")
            continue

        estimators = [
            ["Logistic Regression", LogisticRegression()],
            ["Naive Bayes", GaussianNB()],
            ["SVM", SVC()],
            ["Decision Tree", DecisionTreeClassifier()],
            ["MLP", MLPClassifier()]
            # ["Nearest Neighbors", KNeighborsClassifier(3)],
            # ["Best SVM", SVC()],
            # ["Linear SVM", SVC(kernel="linear", C=0.025)],
            # ["RBF SVM", SVC(gamma=2, C=1)],
            # ["Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)],  # Use bird not cat
            # ["Random Forest", RandomForestClassifier(max_depth=5, n_estimators=1000)],
            # ["Neural Net", MLPClassifier(alpha=1)],
            # ["AdaBoost", AdaBoostClassifier()],
            # ["QDA", QuadraticDiscriminantAnalysis()]
        ]

        for eid, est in enumerate(estimators):
            start_time = time.clock()
            clf = est[1].fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            end_time = time.clock()
            precision, recall, fscore, ts = precision_recall_fscore_support(y_test, y_pred,
                                                                            average='binary',
                                                                            beta=1.0)
            this_time = end_time - start_time
            print("%s: %f, %f, %f, %f" % (est[0], precision, recall, fscore, this_time))
            estimator_score[est[0]].append([y_true_count, precision, recall, fscore, this_time])

    print("Summary for all classes:")
    for est_name, est_arr in estimator_score.items():
        est_prec, est_recall, est_f1, est_time = get_weighted_score(est_arr)
        print("%s: %f %f %f %f" % (est_name, est_prec, est_recall, est_f1, est_time))
