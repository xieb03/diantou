from ..sina.dao import redis_db
from ..sina.dao.mongo_db import MongoDB


class DateRecList(object):
    def __init__(self):
        self._redis = redis_db.Redis()
        self.mongo = MongoDB(db='recommendation')
        self.db_content = self.mongo.db_recommendation
        self.collection_test = self.db_content['content_label']

    def get_news_order_by_time(self):
        ids = list()
        data = self.collection_test.find().sort([{"$news_date", -1}])
        count = 10000

        for news in data:
            self._redis.redis.zadd("rec_date_list", {str(news['_id']): count})
            count -= 1
            print(count)


if __name__ == '__main__':
    date_rec = DateRecList()
    date_rec.get_news_order_by_time()
