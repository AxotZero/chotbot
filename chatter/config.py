import os
from easydict import EasyDict


is_debug = False
gpu = '0'
device = 'cuda'
use_pretrained = True
max_len = 25
max_history_len = 5
candidate_num = 5
repetition_penalty = 2.0
temperature = 1
topk = 8
topp = 0
