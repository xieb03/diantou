# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SinaItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # 标题
    title = scrapy.Field()
    # 内容
    desc = scrapy.Field()
    # 时间
    times = scrapy.Field()
    # 类型
    type = scrapy.Field()

