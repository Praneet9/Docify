from pymongo import MongoClient

def getConnection():
    client = MongoClient('localhost:27017')
    return client

def getDB(client):
    db = client.DocScanner
    return db

def getCollection(collection_name, db):
    collection = db[collection_name]
    return collection

def closeConnection(client):
    client.close()

##Database Helper functions
def insert_data(collection, args_dict):
    client = getConnection()
    db = getDB(client)
    collection_name = getCollection(collection, db)
    '''
    db_name -> string i.e name of the db
    args_dict -> a dictionary of entries in db
    '''
    collection_name.insert_one(args_dict)
    
    closeConnection(client)

def read_data(collection):
    client = getConnection()
    db = getDB(client)
    collection_name = getCollection(collection, db)
    '''
    returns a cursor of objects
    which can be iterated and printed
    '''
    cols = collection_name.find({})
    closeConnection(client)
    return cols

#Update in data base
def update_data(collection, idno, updation):
    client = getConnection()
    db = getDB(client)
    collection_name = getCollection(collection, db)
    '''
    db_name -> string
    idno -> id number of database entry in dict
    '''
    collection_name.update_one(idno, updation)
    closeConnection(client)

def delete_row(collection, idno):
    client = getConnection()
    db = getDB(client)
    collection_name = getCollection(collection, db)
    '''
    Deletes the complete row
    idno must be a dict {idno:'anything'}
    '''
    collection_name.delete_many(idno)
    closeConnection(client)