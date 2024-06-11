import scrapy


class SinaSpiderSpider(scrapy.Spider):
    name = "sina_spider"
    allowed_domains = ["sina.com.cn"]
    start_urls = ["https://sina.com.cn"]

    def parse(self, response, **kwargs):
        pass
