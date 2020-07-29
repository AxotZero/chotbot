import requests
from .utils import *

def response(text):
	# get search url for GoogleSearch
	url = get_google_search_url(text, start_num=0, total_num=5)

	# get html by url
	html = get_google_search_html(url)
	if not html:
		return None

	soup = BeautifulSoup(html, 'lxml')

	# try to get simple description by GoogleSearch result
	answer = get_wiki_description(soup)
	if answer:
		return answer

	# if we didn't get simple description, then we try to get link of Wiki by GoogleSearch result
	link = get_wiki_link(soup)
	if not link:
		return None

	# return the first paragraph of wikipedia page
	return get_wiki_paragraph(link)


def test():
	while True:
		try:
			text = input('question:')
			answer = response(text)
			if answer == None:
				answer = '我不知道ㄟ'
			print('answer:', answer)
		except KeyboardInterrupt:
			break
	return

if __name__ == '__main__':
	test()
