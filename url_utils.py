from bs4 import BeautifulSoup
from bs4.element import Comment, Doctype
import urllib2, ssl
import HTMLParser
class URLUtils:

    @staticmethod
    def clean_text(line):
        clean_line = line.strip()
        clean_line = clean_line.replace('\s+', ' ')
        clean_line = clean_line.replace('\\n+', ' ')
        html_parser = HTMLParser.HTMLParser()
        clean_line = html_parser.unescape(clean_line)
        clean_line = ' '.join(clean_line.split())
        if len(clean_line) <= 1:
            return ''
        # clean_line = clean_line.decode("utf8")
        # print line,str(len(line)) +" =======>"+ clean_line, str(len(clean_line))

        return clean_line


    @staticmethod
    def parse_urls(url):
        print(url)
        context = ssl._create_unverified_context()
        try:
            page = urllib2.urlopen(url, context = context)
        except:
            print 'URL '+url+ " could not be opened"
            return
        soup = BeautifulSoup(page, 'html.parser')
        texts = soup.find_all(text=True)
        filtered_texts = filter(URLUtils.filter_text, texts)
        text = u" ".join(map(URLUtils.clean_text, filtered_texts))
        text = " ".join(text.split())
        return text



    @staticmethod
    def filter_text(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment) or isinstance(element, Doctype):
            return False
        return True
