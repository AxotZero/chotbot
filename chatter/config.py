import os
file_dir = os.path.dirname(os.path.abspath(__file__))

is_debug = False

gpu = '0'
device = 'cuda'
use_pretrained = True
chatlog_path = os.path.join(file_dir, 'chatlog')

max_len = 25
max_history_len = 5
candidate_num = 5
repetition_penalty = 2.0

temperature = 1
topk = 8
topp = 0
