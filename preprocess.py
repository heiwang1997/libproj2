import os
import sys
import pickle
import random

import GetText

if __name__ == '__main__':
    dicpath = sys.argv[1]
    filelist = os.listdir(dicpath)
    fulltextlist = []
    classifierlist = []
    all_files = len(filelist)
    # Random shuffle applied here.
    random.shuffle(all_files)
    print("All files = %d" % all_files)
    for fid, file in enumerate(filelist):
        try:
            fulltext, classifier = GetText.libproj2_get_xml_doc(dicpath + '/' + file)
            print("File No.%d: %s processed\r" % (fid, file), end='')
            fulltextlist.append(fulltext)
            classifierlist.append(classifier)
        except Exception as e:
            print("File No.%d: %s wrong" % (fid, file))
            print(e)
    with open("Saved.pkl", "wb") as f:
        pickle.dump([fulltextlist, classifierlist, filelist], f)
