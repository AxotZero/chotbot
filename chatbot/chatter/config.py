from easydict import EasyDict

debug = True
chatlog_path = './chatlog.txt'

chatter = EasyDict()
chatter.debug = debug
chatter.gpu = '0'
chatter.device = 'cuda'
chatter.use_translator = True
chatter.model_path = 'models/pretrained_model'
chatter.use_mmi = False
chatter.max_len = 25
chatter.max_history_len = 5
chatter.candidate_num = 5
chatter.repetition_penalty = 2.0
chatter.topk = 8
chatter.topp = 0
