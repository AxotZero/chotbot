from answerer.answerer import Answerer
from chatter.chatter import Chatter
import re


def get_question(text):
    
    explain = ['知道', '理解', '解釋', '了解', '瞭解']
    base_q = ['什麼', '甚麼', '誰', '啥']
    q_end = ['阿?', '阿', '啊?', '啊', '嗎?', '嗎', '?']

    text = text.replace('是', '')
    for _end in q_end:
        text = text.replace(_end, '')

	# print(text)
    ex_pattern = '|'.join(explain)
    bq_pattern = '|'.join(base_q)

    # ex bq Q
    pattern = '.*(%s)(%s)(.+)'%(ex_pattern, bq_pattern)
    m = re.match(pattern, text)
    if m and m.group(3) != '':
        return m.group(3)

    # ex Q bq
    pattern = '.*(%s)(.+)(%s)'%(ex_pattern, bq_pattern)
    m = re.match(pattern, text)
    if m and m.group(2) != '':
        return m.group(2)

    # bq Q 
    pattern = '.*(%s)(.+)'%(bq_pattern)
    m = re.match(pattern, text)
    if m and m.group(2) != '':
        return m.group(2)

    # Q bq
    pattern = '(.+)(%s)'%(bq_pattern)
    m = re.match(pattern, text)
    if m and m.group(1) != '':
        return m.group(1)

    # ex Q
    pattern = '.*(%s)(.+)'%(ex_pattern)
    m = re.match(pattern, text)
    if m and m.group(2) != '':
        return m.group(2)

    return ''


class ChatBot():
	def __init__(self):
		self.answerer_bot = Answerer()
		self.chatter_bot = Chatter()		

	def response(self, text):
		question = get_question(text)
		response = ''
		if question:
			response = self.answerer_bot.response(question)
		
		if response == '':
			response = self.chatter_bot.response(text)

		return response


def test():
    chatbot = ChatBot()
    while True:
        try:
            text = input("user:")
            response = chatbot.response(text)
            print('chatbot:', response)

        except KeyboardInterrupt:
            break
    return


if __name__ == "__main__":
	test()
