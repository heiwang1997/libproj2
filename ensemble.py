#!/usr/bin/env python3
# -*- coding=utf-8 -*-

"""
Python script implementing task 3 in project 2, Data Mining.
Comparison of different ensemble methods.
"""

import pickle
import time
from collections import defaultdict

import numpy as np
import scipy.io
import xgboost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


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
    print("Reading Feature file...")
    feat_matrix = scipy.io.mmread("feat.mtx")
    feat_matrix = feat_matrix.tocsr()
    all_document_count = feat_matrix.shape[0]
    train_count = int(all_document_count * 0.9)
    test_count = all_document_count - train_count
    print("Analyse complete. # Valid Document = %d, train = %d, test = %d" %
          (all_document_count, train_count, test_count))

    with open("Class.pkl", "rb") as f:
        [class_name, class_matrix] = pickle.load(f)

    print("Splitting Sparse matrix for training")
    # Shuffle is applied in preprocessing.
    X_train = feat_matrix[0:train_count, :]
    X_test = feat_matrix[train_count:, :]
    class_matrix_train = class_matrix[0:train_count]
    class_matrix_test = class_matrix[train_count:]

    class_name = sorted(list(class_name.items()), key=lambda x: x[1])
    estimator_score = defaultdict(list)

    print("All classes = %d" % len(class_name))
    for [cname, cid] in class_name:
        print("Running Ensembler for class %d - %s" % (cid, cname))
        y_train = get01label(class_matrix_train, cid)
        y_test = get01label(class_matrix_test, cid)

        y_true_count = np.count_nonzero(y_test)
        if y_true_count == 0:
            print("========= NO TRUE SAMPLES ==========")
            continue

        estimators = [
            ["Bootstrap", BaggingClassifier()],
            ["AdaBoost", AdaBoostClassifier()],
            ["Random Forest", RandomForestClassifier()],
            ["Extra Trees", ExtraTreesClassifier()],
            ["KNN Bagging", BaggingClassifier(base_estimator=KNeighborsClassifier())],
            ["NB Boosting", AdaBoostClassifier(base_estimator=MultinomialNB())],
            ["NB Bagging", BaggingClassifier(base_estimator=MultinomialNB())],
            ["XGBoost", None]
        ]

        for eid, est in enumerate(estimators):

            if est[0] == "XGBoost":
                dtrain = xgboost.DMatrix(X_train.tocsc(), label=y_train)
                dtest = xgboost.DMatrix(X_test.tocsc(), label=y_test)
                param = {'max_depth': 6, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic'}
                start_time = time.clock()
                bst = xgboost.train(param, dtrain)
                y_pred = bst.predict(dtest)
                end_time = time.clock()
                y_pred[y_pred > 0.5] = 1
                y_pred[y_pred != 1] = 0
            else:
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
