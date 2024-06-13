from ..sina.dao import redis_db
from ..sina.dao.mongo_db import MongoDB


class MongoToRedisContent(object):
    def __init__(self):
        self._redis = redis_db.Redis()
        self.mongo = MongoDB(db='recommendation')
        self.db_recommendation = self.mongo.db_recommendation
        self.collection_content = self.db_recommendation['content_label']

    def get_from_mongodb(self):
        pipelines = [{
            '$group': {
                '_id': "$type"
            }
        }]
        types = self.collection_content.aggregate(pipelines)

        for type_ in types:
            collection = {"type": type_['_id']}
            data = self.collection_content.find(collection)
            for x in data:
                result = dict()
                result['content_id'] = str(x['_id'])
                result['describe'] = x['describe']
                result['type'] = x['type']
                result['news_date'] = x['news_date']
                result['title'] = x['title']
                self._redis.redis.set("news_detail:" + str(x['_id']), str(result))


if __name__ == '__main__':
    write_to_redis = MongoToRedisContent()
    write_to_redis.get_from_mongodb()
