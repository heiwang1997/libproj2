import pymongo
from bson.objectid import ObjectId
dburi = "mongodb://127.0.0.1"
client = pymongo.MongoClient(dburi)
db = client["data"]
collection = db["data"]


def libproj2_add_doc(docdata):
    result = collection.find_one({"docname": docdata["docname"]})
    if not result:
        id = collection.insert_one(docdata).inserted_id
        return str(id)
    else:
        return "EXIST"


def libproj2_delete_doc_by_name(docname):
    result = collection.delete_many({"docname": docname})
    if result.deleted_count > 1:
        print("server might be error in delete doc")
        return result.deleted_count
    else:
        return result.deleted_count


def libproj2_get_doc_data_by_name(docname):
    result = collection.find_one({"docname": docname})
    return result


def libproj2_get_doc_data(filter):
    result = collection.find(filter)
    s = []
    for element in result:
        s.append(element)
    return s


def libproj2_update_doc_data(filter, updatedata):
    result = collection.update_many(filter, updatedata)
    return result.matched_count


def JTI_initialize_doc():
    collection.delete_many({})