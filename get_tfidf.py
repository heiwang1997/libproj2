from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# import mongotest


def libproj2_get_tfidf(fulltextlist):
    vectorizer = CountVectorizer(min_df=2)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(fulltextlist))
    word = vectorizer.get_feature_names()
    weight = tfidf
    # weight[i][j] means the weight of word j in text i
    return word, weight
