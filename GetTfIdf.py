from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import GetText
import os
# import mongotest


def libproj2_get_tfidf(fulltextlist):
    vectorizer = CountVectorizer(min_df=2)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(fulltextlist))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    # weight[i][j] means the weight of word j in text i
    return word, weight


def libproj2_read_from_db():
    fulltextlist = []
    classifierlist = []
    filenamelist = []
    for doc in mongotest.collection.find():
        fulltextlist.append(doc["fulltext"])
        classifierlist.append(doc["classifier"])
        filenamelist.append(doc["docname"])
    return filenamelist, fulltextlist, classifierlist


if __name__ == '__main__':
    dicpath = 'F:/[3rd year] Spring Semester/Data Mining/samples_50000'
    filelist = os.listdir(dicpath)
    fulltextlist = []
    classifierlist = []
    for file in filelist:
        if mongotest.libproj2_get_doc_data_by_name(file) is None:
            print("Now add file "+file+", Please wait")
            fulltext, classifier = GetText.libproj2_get_xml_doc(dicpath+'/'+file)
            mongotest.libproj2_add_doc({"docname": file, "fulltext": fulltext, "classifier": classifier})
    filenamelist, fulltextlist, classifierlist = libproj2_read_from_db()
    word, weight = libproj2_get_tfidf(fulltextlist)
    print(len(word))
    print(len(weight))