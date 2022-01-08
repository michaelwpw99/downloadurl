import scrapy

url_list = []
class DownloadSpider(scrapy.Spider):
    name = 'download'
    allowed_domains = ['download.cnet.com']
    start_urls = ['https://download.cnet.com/drivers/windows']

    def parse(self, response):
        for link in response.css('.c-productCard_link ::attr(href)'):
            yield response.follow(link.get(), callback=self.parse_links)
        nextpage = response.css('.c-navigationPagination_item--next ::attr(href)').get()
        if nextpage is not None:
            nextpage = response.urljoin(nextpage)
            yield scrapy.Request(nextpage, callback=self.parse)

    def parse_links(self, response):
        buttontext = response.css('.c-productActionButton_text::text').get()
        dllink = response.css('.c-globalButton a::attr(href)').get()
        full_url = response.urljoin(dllink)
        #url_list.append(full_url)

        if buttontext == 'Download Now':
            yield {
                'Button Text': buttontext,
                'Full URL': full_url
            }


