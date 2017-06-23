import os
import pickle
import sys

from scipy.io import mmwrite
from scipy.sparse.csr import csr_matrix

import GetTfIdf

if __name__ == '__main__':
    if not os.path.exists("Saved.pkl"):
        print("No pkl")
        sys.exit(-1)
    print("Loading saved pkl...")
    with open("Saved.pkl", "rb") as f:
        [fulltextlist, classifierlist, filelist] = pickle.load(f)
    all_classes = {}
    classes_count = 0
    p_classifier_list = []
    print("Preprocessing classes...")
    for cl in classifierlist:
        new_cl = []
        for cname in cl:
            if cname in all_classes.keys():
                new_cl.append(all_classes[cname])
            else:
                new_cl.append(classes_count)
                all_classes[cname] = classes_count
                classes_count += 1
        p_classifier_list.append(new_cl)
    with open("Class.pkl", "wb") as f:
        pickle.dump([all_classes, p_classifier_list], f)
    print("Calculating Tf-Idf...")
    word, weight = GetTfIdf.libproj2_get_tfidf(fulltextlist)
    print("Saving Tf-Idf...")
    weight = csr_matrix(weight)
    mmwrite('feat.mtx', weight)
