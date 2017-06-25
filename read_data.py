import pickle

import scipy.io

print("Reading Feature file...")
feat_matrix = scipy.io.mmread("feat.mtx")
# 转换成稀疏矩阵
feat_matrix = feat_matrix.tocsr()
# 总文章数
all_document_count = feat_matrix.shape[0]

with open("Class.pkl", "rb") as f:
    [class_name, class_matrix] = pickle.load(f)
    print(class_name)
    # class_matrix是个二维数据，代表每个文章的分类
    print(class_matrix)