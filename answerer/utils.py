import time
import requests
import urllib.request
import urllib.error
import urllib.parse
from urllib.parse import urlencode
from urllib.parse import parse_qs, urlparse
from fake_useragent import UserAgent
import sys

from bs4 import BeautifulSoup


# PUBLIC

def get_google_search_url(query, start_num=0, total_num=5):
    '''
    Descrpition:
        Get google_search_url for input query.
    Args:
        start_num: int, where you want to start to search.
        total_num: int, how many result you want to search.
    Return:
        url: str, google_search_url
    '''

    params = {
        'nl': 'en',
        'q': query.encode('utf8'),
        'start': start_num,
        'num': total_num
    }

    params = urlencode(params)

    url = u"https://www.google.com/search?" + params

    https = int(time.time()) % 2 == 0
    bare_url = u"https://www.google.com/search?" if https else u"http://www.google.com/search?"
    url = bare_url + params

    return url


def _filter_link(link):
    '''
    Filter links found in the Google result pages HTML code.
    Returns None if the link doesn't yield a valid result.
    '''
    try:
        # Valid results are absolute URLs not pointing to a Google domain
        # like images.google.com or googleusercontent.com
        o = urlparse(link, 'http')
        # link type-1
        # >>> "https://www.gitbook.com/book/ljalphabeta/python-"
        if o.netloc and 'google' not in o.netloc:
            return link
        # link type-2
        # >>> "http://www.google.com/url?url=http://python.jobbole.com/84108/&rct=j&frm=1&q=&esrc=s&sa=U&ved=0ahUKEwj3quDH-Y7UAhWG6oMKHdQ-BQMQFggUMAA&usg=AFQjCNHPws5Buru5Z71wooRLHT6mpvnZlA"
        if o.netloc and o.path.startswith('/url'):
            try:
                link = parse_qs(o.query)['url'][0]
                o = urlparse(link, 'http')
                if o.netloc and 'google' not in o.netloc:
                    return link
            except KeyError:
                pass
        # Decode hidden URLs.
        if link.startswith('/url?'):
            try:
                # link type-3
                # >>> "/url?q=http://python.jobbole.com/84108/&sa=U&ved=0ahUKEwjFw6Txg4_UAhVI5IMKHfqVAykQFggUMAA&usg=AFQjCNFOTLpmpfqctpIn0sAfaj5U5gAU9A"
                link = parse_qs(o.query)['q'][0]
                # Valid results are absolute URLs not pointing to a Google domain
                # like images.google.com or googleusercontent.com
                o = urlparse(link, 'http')
                if o.netloc and 'google' not in o.netloc:
                    return link
            except KeyError:
                # link type-4
                # >>> "/url?url=https://machine-learning-python.kspax.io/&rct=j&frm=1&q=&esrc=s&sa=U&ved=0ahUKEwj3quDH-Y7UAhWG6oMKHdQ-BQMQFggfMAI&usg=AFQjCNEfkUI0RP_RlwD3eI22rSfqbYM_nA"
                link = parse_qs(o.query)['url'][0]
                o = urlparse(link, 'http')
                if o.netloc and 'google' not in o.netloc:
                    return link

    # Otherwise, or on error, return None.
    except Exception:
        pass
    return None


def _get_link(li):
    """
    Return external link from a search.
    """
    try:
        a = li.find("a")
        link = a["href"]
    except Exception:
        return None
    return _filter_link(link)


def get_google_search_html(url):
    '''
    Description:
        Get GoogleSearch HTML by url
    Args:
        url: str, GoogleSearch URL
    Return:
        html: str if success, else None
    '''

    ua = UserAgent()
    header = ua.random
    try:
        request = urllib.request.Request(url)
        request.add_header("User-Agent", header)
        html = urllib.request.urlopen(request).read()
        return html
    except urllib.error.HTTPError as e:
        print("Error accessing:", url)
        print(e)
        if e.code == 503 and 'CaptchaRedirect' in e.read():
            print("Google is requiring a Captcha. "
                  "For more information check: 'https://support.google.com/websearch/answer/86640'")
        if e.code == 503:
            sys.exit("503 Error: service is currently unavailable. Program will exit.")
        return None
    except Exception as e:
        print("Error accessing:", url)
        print(e)
        return None


def get_wiki_link(soup):
    '''
    Description:
        Get link of wiki from GoogleSearch HTML.
    Args:
        soup: BeautifulSoup object
    Return:
        link: str if find wiki link, else None
    '''

    divs = soup.findAll("div", attrs={"class": "g"})
    results_div = soup.find("div", attrs={"id": "resultStats"})

    links = []
    for li in divs:
        link = _get_link(li)
        if link and ('wikipedia.org' in link):
            return link

    return None


def get_wiki_paragraph(url):
    '''
    Description:
        Get first paragraph from wikipage
    Args:
        url: str, url of wikipage
    Return:
        answer: str if success, else None
    '''

    headers = {'User-Agent': 'Googlebot', 'From': 'YOUR EMAIL ADDRESS' }
    pageSource = requests.get(url, headers=headers).text

    try:
        answer = BeautifulSoup(pageSource, 'lxml').find('div', class_='mw-parser-output').find('p')
        for tag in answer.find_all('sup'):
            tag.replaceWith('')

        return answer.text
    except:
        return None


def get_wiki_description(soup):
    '''
    Description:
        Get simple description from google search wiki result.
    Args:
        soup: BeautifulSoup object
    Return:
        answer: str if success, else None
    '''
    try:
        answer = soup.find('div', class_='kno-rdesc').find('span').text
        return answer
    except:
        return None


def test():
    url = get_google_search_url('迪麗熱巴')
    html = get_google_search_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, 'lxml')

    print('Simple description from google search:')
    print(get_wiki_description(soup))


    link = get_wiki_link(soup)

    print('First Pragraph of wiki:')
    print(get_wiki_paragraph(link))


if __name__ == "__main__":
	test()

