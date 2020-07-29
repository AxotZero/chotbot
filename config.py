import os
from easydict import EasyDict

chatlog_path = './chatlog.txt'


chatter = EasyDict()
chatter.is_debug = False
chatter.gpu = '0'
chatter.device = 'cuda'
chatter.use_pretrained = True
chatter.max_len = 25
chatter.max_history_len = 5
chatter.candidate_num = 5
chatter.repetition_penalty = 2.0
chatter.temperature = 1
chatter.topk = 8
chatter.topp = 0
