from pymongo import MongoClient
import pprint
import re


client = MongoClient('mongodb://localhost:27017/')  

db = client.chatbot

collection = db.managers

# db.managers.find({ "name": { $exists: true, $ne: null } })
def get_manager(group,year):
    return list(collection.find({"name":group  ,"year":year}))[0]["manager"]
    #re.compile(group, re.IGNORECASE)
    





