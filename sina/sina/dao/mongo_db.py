import pymongo


class MongoDB(object):
    def __init__(self, db):
        mongo_client = self._connect('127.0.0.1', 27017, '', '', db)
        self.db_scrapy = mongo_client['scrapy_data']
        self.db_recommendation = mongo_client['db_recommendation']
        self.db_client = mongo_client['db_client']
        self.collection_test = self.db_scrapy['test_collections']

    def _connect(self, host, port, user, pwd, db):
        mongo_info = self._splicing(host, port, user, pwd, db)
        mongo_client = pymongo.MongoClient(mongo_info, connectTimeoutMS=12000, connect=False)
        return mongo_client

    @staticmethod
    def _splicing(host, port, user, pwd, db):
        client = 'mongodb://' + host + ":" + str(port) + "/"
        if user != '':
            client = 'mongodb://' + user + ':' + pwd + '@' + host + ":" + str(port) + "/"
            if db != '':
                client += db
        return client
