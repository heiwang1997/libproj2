import xmltodict
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
# when first use it
# must run nltk.download() to initialize


def libproj2_is_true_type(typetext):
    typelist = typetext.split('/')
    if len(typelist) >= 3:
        if typelist[0] == 'Top':
            if typelist[1] == 'News' or typelist[1] == 'Features':
                return True
    return False


def libproj2_get_true_type(typetext):
    typelist = typetext.split('/')
    if len(typelist) >= 3:
        if typelist[0] == 'Top':
            if typelist[1] == 'News' or typelist[1] == 'Features':
                return typelist[2]
    return ''


def libproj2_pretreatment_doc(fulltext, classifier):
    st = LancasterStemmer()
    wordlist = word_tokenize(fulltext)
    wordlist = [word.lower() for word in wordlist]
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '--']
    wordlist = [word for word in wordlist if word not in english_punctuations]
    wordlist = [w for w in wordlist if w not in stopwords.words('english')]
    wordlist = [w for w in wordlist if w.isdigit() is False]
    wordlist = [st.stem(word) for word in wordlist]
    classifier = list(set([libproj2_get_true_type(w) for w in classifier if libproj2_is_true_type(w)]))
    result = ""
    for word in wordlist:
        result += word + ' '
    return result, classifier


def libproj2_get_xml_doc(filepath):
    file1 = open(filepath, 'r', encoding='utf-8')
    fileinput = file1.read()
    file1.close()
    fulldata = xmltodict.parse(fileinput)['nitf']
    fulltext = ''
    classifier = []
    if fulldata is not None and 'body' in fulldata.keys():
        data = fulldata['body']
        if data is not None and 'body.content' in data.keys():
            data = data['body.content']
            if data is not None and 'block' in data.keys():
                data = data['block']
                if type(data) == list:
                    for element in data:
                        if element is not None and '@class' in element.keys():
                            if element['@class'] == 'full_text':
                                datalist = element['p']
                                if (type(datalist) == str):
                                    fulltext = datalist
                                else:
                                    res = ""
                                    for line in datalist:
                                        res += line + ' '
                                    fulltext = res
                else:
                    element = data
                    if element is not None and '@class' in element.keys():
                        if element['@class'] == 'full_text':
                            datalist = element['p']
                            if (type(datalist) == str):
                                fulltext = datalist
                            else:
                                res = ""
                                for line in datalist:
                                    res += line + ' '
                                fulltext = res
    if fulldata is not None and 'head' in fulldata.keys():
        data = fulldata['head']
        if data is not None and 'docdata' in data.keys():
            data = data['docdata']
            if data is not None and 'identified-content' in data.keys():
                data = data['identified-content']
                if data is not None and 'classifier' in data.keys():
                    data = data['classifier']
                    if type(data) == list:
                        for element in data:
                            classifier.append(element['#text'])
                    else:
                        classifier.append(data['#text'])
    return libproj2_pretreatment_doc(fulltext, classifier)


if __name__ == '__main__':
    wordlist, classifier = libproj2_get_xml_doc("C:/Users/zjkgf/Desktop/55990477_5_project2/samples_50000/1855667.xml")
    wordlist, classifier = libproj2_get_xml_doc("C:/Users/zjkgf/Desktop/55990477_5_project2/samples_50000/1802525.xml")