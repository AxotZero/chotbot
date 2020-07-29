import requests
from bs4 import BeautifulSoup
from google import google




def _get_wiki_url(text):
	search_result = google.search(text, 1)
	print('search_result:')
	print(search_result)
	for result in search_result:
		if result.link and ('wikipedia.org' in result.link):
			return result.link

	return ''


def _get_wiki_paragraph(url):
	headers = {'User-Agent': 'Googlebot', 'From': 'YOUR EMAIL ADDRESS' }
	pageSource = requests.get(url, headers=headers).text

	answer = BeautifulSoup(pageSource, 'lxml').find('div', class_='mw-parser-output').find('p')
	print('answer:', answer)
	for tag in answer.find_all('sup'):
	    tag.replaceWith('')

	return answer.text


def _get_wiki_description(text):
	desc = google.get_wiki_description(text)
	return desc


def response(text):
	desc = ''
	# desc = _get_wiki_description(text)
	if desc != '':
		return desc

	url = _get_wiki_url(text)
	if url == '':
		return ''
	else:
		return _get_wiki_paragraph(url)


def test():
	while True:
		try:
			text = input('question:')
			answer = response(text)
			if answer == '':
				answer = '我不知道ㄟ'
			print('answer:')
			print(answer)
		except KeyboardInterrupt:
			break
	return

if __name__ == '__main__':
	test()
