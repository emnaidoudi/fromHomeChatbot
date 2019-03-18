from pymongo import MongoClient
import pprint


client = MongoClient('mongodb://localhost:27017/')  

db = client.chatbot

collection = db.groups

def get_manager(group,year):
    print(year)
    return list(collection.find({"name":group,"year":year}))[0]["manager"]




