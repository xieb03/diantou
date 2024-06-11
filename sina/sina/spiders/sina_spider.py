import datetime
import re
from typing import Any

import scrapy
from scrapy.http import Request, Response
from scrapy.selector import Selector
from selenium import webdriver
from selenium.webdriver.common.by import By

from ..items import SinaItem


class SinaSpiderSpider(scrapy.Spider):
    name = "sina_spider"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.start_urls = ['https://news.sina.com.cn/china/']
        self.option = webdriver.ChromeOptions()
        # 取消沙盒模式，也就是说让它在 root 权限下执行
        self.option.add_argument('no=sandbox')
        # 不加载图片，因为我们只想要文字部分，加上这一句可以提升爬取速度和效率。
        self.option.add_argument('--blink-setting=imagesEnable=false')

    # 定义 Scrapy 框架的起始请求，如果在这个起始请求中有重复的 URL，它会自动进行去重操作。
    def start_requests(self):
        for url in self.start_urls:
            yield Request(url=url, callback=self.parse)

    # 这个解析函数的入参就是服务器返回的 response 值
    def parse(self, response: Response, **kwargs: Any):
        driver = webdriver.Chrome(options=self.option)
        # 我们还把加载页面的超时时间设置为了 30 秒，也就是说如果 30 秒还加载不出来，就去请求下一个页面，而这下一个页面就是从 start_requests() 函数中获得的
        driver.set_page_load_timeout(30)
        driver.get(response.url)

        for index in range(2):
            print(F"-------------------第 {index} 页")
            # 一直下滑到底部
            while not driver.find_element(by=By.XPATH, value="//div[@class='feed-card-page']").text:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            title = driver.find_elements(by=By.XPATH,
                                         value="//h2[@class='undefined' or @class='feed-card-icon feed-card-icon-hot  undefined']/a[@target='_blank']")
            time = driver.find_elements(
                by=By.XPATH,
                value="//h2[@class='undefined' or @class='feed-card-icon feed-card-icon-hot  undefined']/../div[@class='feed-card-a " "feed-card-clearfix']/div[@class='feed-card-time']")
            for i in range(min(5, len(title))):
                print(title[i].text)
                print(time[i].text)

                today = datetime.datetime.now()
                eachtime = time[i].text
                eachtime = eachtime.replace('今天', str(today.month) + '月' + str(today.day) + '日')

                if '分钟前' in eachtime:
                    minute = int(eachtime.split('分钟前')[0])
                    t = datetime.datetime.now() - datetime.timedelta(minutes=minute)
                    t2 = datetime.datetime(year=t.year, month=t.month, day=t.day, hour=t.hour, minute=t.minute)
                else:
                    if '年' not in eachtime:
                        eachtime = str(today.year) + '年' + eachtime
                    t1 = re.split('[年月日:]', eachtime)
                    t2 = datetime.datetime(year=int(t1[0]), month=int(t1[1]), day=int(t1[2]), hour=int(t1[3]),
                                           minute=int(t1[4]))

                print(t2)

                href = title[i].get_attribute('href')
                print(href)

                item = SinaItem()
                item['type'] = 'news'
                item['title'] = title[i].text
                item['times'] = t2

                # 这个 yield 出去的 Item 实际上会被传入到 Item Pipeline 中，而这个 Item Pipeline 实际上就是对应着 pipelines.py 文件。
                yield Request(url=response.urljoin(href), meta={'name': item}, callback=self.parse_detail)

            # noinspection PyBroadException
            try:
                driver.find_element(by=By.XPATH,
                                    value="//div[@class='feed-card-page']/span[@class='pagebox_next']/a").click()
            except Exception:
                break

    @staticmethod
    def parse_detail(response):
        # 我们首先建立了一个 Selector，它主要是 Response 用来提取数据的。
        # 当 Spider 的 Request 得到 Response 之后，Spider 可以使用 Selector 提取 Response 中的有用的数据。因此，这里我们传入的是上面的 Response 信息，也就是详情页的 Response。
        selector = Selector(response)
        desc = selector.xpath("//div[@class='article']/p/text()").extract()
        item = response.meta['name']
        desc = list(map(str.strip, desc))
        item['desc'] = ''.join(desc)
        print(item)
        yield item

# def parse(self, response):
#     driver = webdriver.Chrome(chrome_options=self.option)
#     driver.set_page_load_timeout(30)
#     driver.get(response.url)
#
#     for i in range(5):
#         while not driver.find_element_by_xpath("//div[@class='feed-card-page']").text:
#             driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         title = driver.find_elements_by_xpath("//h2[@class='undefined']/a[@target='_blank']")
#         time = driver.find_elements_by_xpath(
#             "//h2[@class='undefined']/../div[@class='feed-card-a feed-card-clearfix']/div[@class='feed-card-time']")
#         for i in range(len(title)):
#             print(title[i].text)
#             print(time[i].text)
#
#             today = datetime.datetime.now()
#             eachtime = time[i].text
#             eachtime = eachtime.replace('今天', str(today.month) + '月' + str(today.day) + '日')
#
#             href = title[i].get_attribute('href')
#
#             if '分钟前' in eachtime:
#                 minute = int(eachtime.split('分钟前')[0])
#                 t = datetime.datetime.now() - datetime.timedelta(minutes=minute)
#                 t2 = datetime.datetime(year=t.year, month=t.month, day=t.day, hour=t.hour, minute=t.minute)
#             else:
#                 if '年' not in eachtime:
#                     eachtime = str(today.year) + '年' + eachtime
#                 t1 = re.split('[年月日:]', eachtime)
#                 t2 = datetime.datetime(year=int(t1[0]), month=int(t1[1]), day=int(t1[2]), hour=int(t1[3]),
#                                        minute=int(t1[4]))
#
#             print(t2)
#
#             item = SinaItem()
#             item['type'] = 'news'
#             item['title'] = title[i].text
#             item['times'] = t2
#
#             yield Request(url=response.urljoin(href), meta={'name': item}, callback=self.parse_namedetail)
#
#         try:
#             driver.find_element_by_xpath("//div[@class='feed-card-page']/span[@class='pagebox_next']/a").click()
#         except:
#             Break
