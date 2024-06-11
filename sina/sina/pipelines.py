# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# noinspection PyUnresolvedReferences
from itemadapter import ItemAdapter

# noinspection PyUnresolvedReferences
from .dao.mongo_db import MongoDB


class SinaPipeline:
    def __init__(self):
        # self.mongo = MongoDB(db='scrapy_data')
        # self.collection = self.mongo.db_scrapy['content_ori']
        self.count = 0
        pass

    # 将 pipeline 和爬虫程序进行关联，这个关联的操作在 settings.py 文件中进行。
    # # Configure item pipelines
    # # See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
    # ITEM_PIPELINES = {
    #    "sina.pipelines.SinaPipeline": 300,
    # }
    def process_item(self, item, spider):
        self.count += 1
        print(F"已经下载了 {self.count} 篇文章.")
        # 数据转换成 MongoDB 所需要的 bson 类型，然后插入即可。
        # MongoDB 中的 bson 类型实际上和 Python 中的 dict 格式一样，因此，我们就将 Item 转换成 dict 类型，并赋值给 result_item 这个变量，
        # 然后调用 self.collection.insert_one() 这个函数来进行数据的插入。
        # result_item = dict(item)
        # self.collection.insert_one(result_item)
        return item
