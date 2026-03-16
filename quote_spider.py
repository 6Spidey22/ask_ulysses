import scrapy

class QuoteSpider(scrapy.Spider):
    name = "quotes"
    start_urls = []

    def set_start_urls(self, start_urls):
        self.start_urls = start_urls

    def parse(self, response):
        # Extract quotes and authors using CSS selectors
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').get(),
                'author': quote.css('small.author::text').get(),
                'tags': quote.css('div.tags a.tag::text').getall(),
            }

        # Follow the link to the next page
        next_page = response.css('li.next a::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
