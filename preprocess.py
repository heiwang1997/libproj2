import os
import sys
import pickle
import random

import get_text

if __name__ == '__main__':
    dicpath = sys.argv[1]
    filelist = os.listdir(dicpath)
    fulltextlist = []
    classifierlist = []
    all_files = len(filelist)
    # Random shuffle applied here.
    random.shuffle(filelist)
    print("All files = %d" % all_files)
    for fid, file in enumerate(filelist):
        try:
            fulltext, classifier = get_text.libproj2_get_xml_doc(dicpath + '/' + file)
            if not classifier:
                print("Omitting file %d: %s with no class" % (fid, file))
                continue
            print("File No.%d: %s processed\r" % (fid, file), end='')
            fulltextlist.append(fulltext)
            classifierlist.append(classifier)
        except Exception as e:
            print("File No.%d: %s wrong" % (fid, file))
            print(e)
    with open("Saved.pkl", "wb") as f:
        pickle.dump([fulltextlist, classifierlist, filelist], f)
