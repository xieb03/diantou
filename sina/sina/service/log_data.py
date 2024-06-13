from datetime import datetime

from ..dao import mongo_db
from ..dao import redis_db


# 日志行为
class LogData(object):
    def __init__(self):
        self._mongo = mongo_db.MongoDB(db='recommendation')
        self._redis = redis_db.Redis()

    # 用户行为记录在 mongodb 中
    # 这里的表名主要是前面传过来用来区分是点赞还是收藏的字段，当前端传过来的是 likes 时，我们就会向 likes 表里插入一条记录点赞信息的记录，
    # 同理如果前端传过来的是 collections，就会向 collections 表里增加一条收藏的记录。
    def insert_log(self, user_id, content_id, title, table):
        # noinspection PyBroadException
        try:
            collection = self._mongo.db_client[table]
            info = dict()
            info['user_id'] = user_id
            info['content_id'] = content_id
            info['title'] = title
            info['date'] = datetime.utcnow()
            collection.insert_one(info)
            return True
        except Exception:
            return False

    # 用于修改 Redis 中的一些日志信息
    def modify_article_detail(self, key, ops):
        # noinspection PyBroadException
        try:
            info = self._redis.redis.get(key)
            info = eval(info)
            info[ops] += 1
            self._redis.redis.set(key, str(info))

            return True
        except Exception:
            return False
