#!/usr/bin/env python3
# -*- coding=utf-8 -*-

"""
Python script implementing task 4 in project 2, Data Mining.
Clustering Algorithms
"""

import pickle
import time
import scipy.io
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.cluster import adjusted_mutual_info_score, normalized_mutual_info_score

if __name__ == '__main__':

    print("Reading Feature file...")
    feat_matrix = scipy.io.mmread("feat.mtx")
    feat_matrix = feat_matrix.tocsr()

    all_document_count = feat_matrix.shape[0]

    with open("Class.pkl", "rb") as f:
        [class_name, class_matrix] = pickle.load(f)

    svd = TruncatedSVD(n_components=200)
    res = svd.fit_transform(feat_matrix)
    res *= 100
    print(res)

    dbscan = DBSCAN(eps=13)
    dbscan2 = DBSCAN(eps=15)
    print(dbscan)
    print(dbscan2)

    print(time.time())
    predictdata = dbscan.fit_predict(res)
    print(time.time())
    predictdata2 = dbscan2.fit_predict(feat_matrix)

    svdpac = TruncatedSVD(n_components=2)
    svdres = svdpac.fit_transform(res)
    svdres *= 100

    tempX = []
    tempY = []
    colornum = []
    colorbar = []
    colorname = []
    for name, hex in matplotlib.colors.cnames.items():
        colorname.append(hex)
    for j in range(0, max(predictdata)):
        num = 0
        for i in range(0, len(predictdata)):
            if predictdata[i] == j:
                tempX.append(svdres[i][0])
                tempY.append(svdres[i][1])
                num += 1
        colornum.append(num)
    for j in range(0, len(class_name)):
        for i in range(0, colornum[j]):
            colorbar.append(colorname[j])
    plt.figure(figsize=(20, 20), dpi=80)
    plt.scatter(tempX, tempY, c=colorbar)
    plt.show()

    svdpac = TruncatedSVD(n_components=2)
    svdres = svdpac.fit_transform(res)
    svdres *= 100

    tempX = []
    tempY = []
    colornum = []
    colorbar = []
    colorname = []
    for name, hex in matplotlib.colors.cnames.items():
        colorname.append(hex)
    for j in range(0, max(predictdata2)):
        num = 0
        for i in range(0, len(predictdata2)):
            if predictdata2[i] == j:
                tempX.append(svdres[i][0])
                tempY.append(svdres[i][1])
                num += 1
        colornum.append(num)
    for j in range(0, len(class_name)):
        for i in range(0, colornum[j]):
            colorbar.append(colorname[j])
    plt.figure(figsize=(20, 20), dpi=80)
    plt.scatter(tempX, tempY, c=colorbar)
    plt.show()

    true_result = []
    for element in class_matrix:
        true_result.append(element[0])
    print(true_result)
    print(adjusted_mutual_info_score(predictdata, true_result))
    print(normalized_mutual_info_score(predictdata, true_result))
    print(adjusted_mutual_info_score(predictdata2, true_result))
    print(normalized_mutual_info_score(predictdata2, true_result))

    kmeans = KMeans(n_clusters=max(predictdata) + 1)
    print(kmeans)
    print(time.time())
    kmeanspredictdata = kmeans.fit_predict(res)
    print(time.time())
    kmeans2 = KMeans(n_clusters=max(predictdata2) + 1)
    print(kmeans2)
    kmeanspredictdata2 = kmeans2.fit_predict(feat_matrix)

    svdpac = TruncatedSVD(n_components=2)
    svdres = svdpac.fit_transform(res)
    svdres *= 100
    tempX = []
    tempY = []
    colornum = []
    colorbar = []
    colorname = []
    for name, hex in matplotlib.colors.cnames.items():
        colorname.append(hex)
    for j in range(0, max(kmeanspredictdata)):
        num = 0
        for i in range(0, len(kmeanspredictdata)):
            if kmeanspredictdata[i] == j:
                tempX.append(svdres[i][0])
                tempY.append(svdres[i][1])
                num += 1
        colornum.append(num)
    for j in range(0, len(class_name)):
        for i in range(0, colornum[j]):
            colorbar.append(colorname[j])
    plt.figure(figsize=(20, 20), dpi=80)
    plt.scatter(tempX, tempY, c=colorbar)
    plt.show()

    svdpac = TruncatedSVD(n_components=2)
    svdres = svdpac.fit_transform(res)
    svdres *= 100

    tempX = []
    tempY = []
    colornum = []
    colorbar = []
    colorname = []
    for name, hex in matplotlib.colors.cnames.items():
        colorname.append(hex)
    for j in range(0, max(kmeanspredictdata2)):
        num = 0
        for i in range(0, len(kmeanspredictdata2)):
            if kmeanspredictdata[i] == j:
                tempX.append(svdres[i][0])
                tempY.append(svdres[i][1])
                num += 1
        colornum.append(num)
    for j in range(0, len(class_name)):
        for i in range(0, colornum[j]):
            colorbar.append(colorname[j])
    plt.figure(figsize=(20, 20), dpi=80)
    plt.scatter(tempX, tempY, c=colorbar)
    plt.show()

    true_result = []
    for element in class_matrix:
        true_result.append(element[0])
    print(true_result)
    print(adjusted_mutual_info_score(kmeanspredictdata, true_result))
    print(normalized_mutual_info_score(kmeanspredictdata, true_result))
    print(adjusted_mutual_info_score(kmeanspredictdata2, true_result))
    print(normalized_mutual_info_score(kmeanspredictdata2, true_result))
