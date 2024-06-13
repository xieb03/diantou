from ..sina.dao import redis_db


class PageUtils(object):
    def __init__(self):
        self._redis = redis_db.Redis()

    # 这段代码主要是实现一个翻页的功能
    def get_data_with_page(self, page_num, page_size):
        start = (page_num - 1) * page_size
        end = start + page_size
        # Return a range of values from sorted set ``name`` between ``start`` and ``end`` sorted in descending order.
        data = self._redis.redis.zrevrange("rec_date_list", start, end)
        lst = list()
        for x in data:
            info = self._redis.redis.get("news_detail:" + x)
            lst.append(info)
        return lst


if __name__ == '__main__':
    page_utils = PageUtils()
    print(page_utils.get_data_with_page(1, 20))
