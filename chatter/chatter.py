import os
from os.path import join, exists
import copy
import logging
from easydict import EasyDict

import torch
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F
from googletrans import Translator


file_dir = os.path.dirname(os.path.abspath(__file__))


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


class Chatter():
    def __init__(self, config):
        logging.info('Chatter Start Initializing.')

        self.args = self._process_config(config)
        self.device = self._get_device()
        self.tokenizer = self._get_tokenizer()
        self.dialogue_model = self._get_dialogue_model()
        self.mmi_model = self._get_mmi_model()
        self.translator = self._get_translator()
        self.history = []

        logging.info('Chatter Finish Initializing.')


    def _process_config(self, config):

        args = EasyDict()

        args.debug = config.get('debug', False)

        args.gpu = config.get('gpu', '')

        args.device = config.get('device', 'cpu')

        args.max_len = config.get('max_len', 25)

        args.max_history_len = config.get('max_history_len', 5)

        args.candidate_num = config.get('candidate_num', 5)

        args.repetition_penalty = config.get('repetition_penalty', 2)

        args.temperature = config.get('temperature', 1)

        args.topk = config.get('topk', 8)

        args.topp = config.get('topp', 0)

        args.use_translator = config.get('use_translator', False)
        
        args.dialogue_model_path = join(file_dir, config.model_path, 'dialogue')
        args.mmi_model_path = join(file_dir, config.model_path, 'mmi')
        args.vocab_path = join(file_dir, config.model_path, 'vocab.txt')
        
        logging.info('Finish Processing Config.')

        return args


    def _get_device(self):
        args = self.args
        # set visible gpu.
        if args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        # check if gpu is available. 
        gpu_available = torch.cuda.is_available()
        if args.gpu and gpu_available:
            logging.info("Using gpu: %s" % (args.gpu))
        else:
            logging.info("Using cpu.")

        # return used device
        if gpu_available and args.device:
            return args.device
        else:
            return 'cpu'


    def _get_tokenizer(self):
        args = self.args
        logging.info("Start getting tokenizer ")

        tokenizer = BertTokenizer(vocab_file=args.vocab_path)

        logging.info("Finish getting tokenizer ")

        return tokenizer


    def _get_dialogue_model(self):
        logging.info("Start getting dialogue model.")

        args = self.args
        dialogue_model = GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
        dialogue_model.to(self.device)
        dialogue_model.eval()
        
        logging.info("Finish reading dialogue model.")

        return dialogue_model


    def _get_mmi_model(self):
        logging.info("Start getting mmi model.")

        args = self.args
        mmi_model = GPT2LMHeadModel.from_pretrained(args.mmi_model_path)
        mmi_model.to(self.device)
        mmi_model.eval()

        logging.info("Finish getting mmi model.")

        return mmi_model


    def _get_translator(self):
        args = self.args
        if args.use_translator:
            logging.info("Using Translator.")
            return Translator()
        else:
            return None


    def response(self, text):
        if self.translator:
            text = self.translator.translate(text, dest='zh-cn').text

        input_ids = self.get_input_ids(text)
        candidate_response = self.get_candidate_response(input_ids)
        candidate_response = self.candidate_response_filter(candidate_response)
        response = self.select_response(candidate_response)
        text = self.tokenizer.convert_ids_to_tokens(response)
        text = "".join(text)

        if self.translator:
            text = self.translator.translate(text, dest='zh-tw').text

        return text


    def update_history_text(self, text):
        args = self.args

        if len(text) > args.max_len:
            text = text[:args.max_len]

        ids = self.tokenizer.encode(text)
        self.history.append(ids)

        if len(self.history) > args.max_history_len:
            self.history = self.history[len(self.history)-args.max_history_len:]


    def update_history_id(self, ids):
        args = self.args

        if len(ids) > args.max_len:
            ids = ids[:args.max_len]

        self.history.append(ids)

        if len(self.history) > args.max_history_len:
            self.history = self.history[len(self.history)-args.max_history_len:]


    def get_input_ids(self, text):
        args = self.args

        self.update_history_text(text)

        # 每个input以[CLS]为开头
        input_ids = [self.tokenizer.cls_token_id]  
        for history_id, history_utr in enumerate(self.history[-args.max_history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(self.tokenizer.sep_token_id)

        # 用于批量生成response，维度为(candidate_num,token_len)
        input_ids = [copy.deepcopy(input_ids) for _ in range(args.candidate_num)]
        input_ids = torch.tensor(input_ids).long().to(self.device)
        return input_ids


    def get_candidate_response(self, input_ids):
        args = self.args

        generated = []  # 二维数组，维度为(生成的response的最大长度，candidate_num)，generated[i,j]表示第j个response的第i个token的id
        finish_set = set()  # 标记是否所有response均已生成结束，若第i个response生成结束，即生成了sep_token_id，则将i放入finish_set
        
        # 最多生成max_len个token
        for _ in range(args.max_len):

            outputs = self.dialogue_model(input_ids=input_ids)
            next_token_logits = outputs[0][:, -1, :]

            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for index in range(args.candidate_num):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature

            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            for next_token_logit in next_token_logits:
                next_token_logit[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)

            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            # 判断是否有response生成了[SEP],将已生成了[SEP]的resposne进行标记
            for index, token_id in enumerate(next_token[:, 0]):
                if token_id == self.tokenizer.sep_token_id:
                    finish_set.add(index)
            
            # 检验是否所有的response均已生成[SEP]
            finish_flag = True  # 是否所有的response均已生成[SEP]的token
            for index in range(args.candidate_num):
                if index not in finish_set:  # response批量生成未完成
                    finish_flag = False
                    break
            if finish_flag:
                break
            generated.append([token.item() for token in next_token[:, 0]])
            
            # 将新生成的token与原来的token进行拼接
            input_ids = torch.cat((input_ids, next_token), dim=-1)

        candidate_responses = []  # 生成的所有候选response
        for batch_index in range(args.candidate_num):
            response = []
            for token_index in range(len(generated)):
                if generated[token_index][batch_index] != self.tokenizer.sep_token_id:
                    response.append(generated[token_index][batch_index])
                else:
                    break
            candidate_responses.append(response)

        return candidate_responses


    def candidate_response_filter(self, candidate_response):
        banned_list = []
        pop_list = []
        translator = Translator()
        banned_words = ['圖片評論', '屎', '傻逼', '智障']
        for i, response in enumerate(candidate_response):
            text = ''.join(self.tokenizer.convert_ids_to_tokens(response))
            text = translator.translate(text, dest='zh-tw').text

            if text in self.history:
                pop_list.append(i)

            for banned_word in banned_words:
                if banned_word in text:
                    candidate_response.pop(i)
                    break

        if len(pop_list) != len(candidate_response):
            for i in pop_list:
                candidate_response.pop(i)
        
        return candidate_response


    def select_response(self, candidate_response):
        args = self.args

        loss_list = []
        reverse_history = reversed(self.history[-args.max_history_len:])

        for response in candidate_response:
            mmi_input_id = [self.tokenizer.cls_token_id]  # 每个input以[CLS]为开头
            mmi_input_id.extend(response)
            mmi_input_id.append(self.tokenizer.sep_token_id)

            for history_utr in reverse_history:
                mmi_input_id.extend(history_utr)
                mmi_input_id.append(self.tokenizer.sep_token_id)
            mmi_input_tensor = torch.tensor(mmi_input_id).long().to(self.device)
            out = self.mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)
            loss = out[0].item()

            if args.debug:
                text = self.tokenizer.convert_ids_to_tokens(response)
                logging.info("{} loss:{}".format("".join(text), loss))

            loss_list.append(loss)


        best_response = ""
        pairs = sorted(zip(candidate_response, loss_list), key=lambda s: s[1])
        for response, _ in pairs:
            if response in reverse_history:
                continue
            else:
                best_response = response
                break 

        if best_response == "":
            best_response = pairs[0][0]

        self.update_history_id(best_response)
        return best_response


def test():
    import config
    if config.debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    chatbot = Chatter(config.chatter)
    while True:
        try:
            text = input("user:")
            res = chatbot.response(text)
            print('chatbot:', res)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    test()