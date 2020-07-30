import re
import os
from os.path import join
from datetime import datetime
import logging

from answerer import answerer
from chatter.chatter import Chatter

import config


def logging_setting():
	if config.debug:
		logging.basicConfig(level=logging.INFO)
	else:
		logging.basicConfig(level=logging.WARNING)


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
		self.chatter_bot = Chatter(config.chatter)
		self.chatlog_writer = self._get_chatlog_writer()


	def __del__(self):
		self.write_chatlog("------對話結束------\n")
		self.write_chatlog("\n")

		if self.chatlog_writer:
			self.chatlog_writer.close()


	def _get_chatlog_writer(self):
		if not config.chatlog_path:
			return None
		else:
			chatlog_writer = open(config.chatlog_path, 'a+', encoding='utf-8')
			chatlog_writer.write("---聊天開始於 {} :---\n".format(datetime.now()))
			return chatlog_writer 


	def write_chatlog(self, text):
		if self.chatlog_writer:
			self.chatlog_writer.write(text + '\n')


	def response(self, text):
		self.write_chatlog('user: '+ text)

		question = get_question(text)
		response = None
		if question:
			response = answerer.response(question)
		
		if not response:
			response = self.chatter_bot.response(text)
			logging.info('chatter: ' + response)
		else:
			self.chatter_bot.update_history_text(response)
			logging.info('answerer: ' + response)

		self.write_chatlog('chatbot: '+ response)
		return response



def test():
	logging_setting()
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
